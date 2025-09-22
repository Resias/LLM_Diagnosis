import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Subset
import torch.optim as optim
import torch.multiprocessing as mp
import glob

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from functools import partial

from data.dataset import WindowedVibrationDataset, OrderInvariantSignalImager
from sklearn.metrics import precision_score, recall_score, f1_score
from tokenizer_trainer.visualize import create_reconstruction_figure
import matplotlib.pyplot as plt

from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE

import wandb
import ast
from tqdm import tqdm
import argparse
import types

def unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model

def save_checkpoint(save_path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int, meta: dict=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    net = unwrap_ddp(model)
    payload = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if meta:
        payload['meta'] = meta
    torch.save(payload, save_path)

def _ddp_sum_tensor(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def train_model(alpha, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, rank, config, start_epoch=0):
    net = unwrap_ddp(model)  # <-- 추가

    best_val_loss = float("inf")
    is_main_process = rank == 0  # 메인 프로세스 여부 확인
    warmup_epochs = getattr(config, "warmup_epochs", 0)   # 새 인자 사용
    save_dir = getattr(config, "save_dir", "checkpoints")
    save_every = int(getattr(config, "save_every", 0))
    max_keep = int(getattr(config, "max_keep", 5))

    for epoch in range(start_epoch, num_epochs):
        LOG_EMBED_INTERVAL = 50  # 고정 주기. 0이면 비활성화, >0이면 해당 주기마다만 로깅
        should_log_embed = (
            is_main_process and wandb.run is not None and
            LOG_EMBED_INTERVAL > 0 and
            ((epoch == 0) or ((epoch + 1) % LOG_EMBED_INTERVAL == 0))
        )

        # --- (핵심) 에폭별 α 결정: 워밍업 구간에서는 분류 손실 비중 0 ---
        if getattr(config, "recon_only", False):
            effective_alpha = 0.0
            phase_name = "recon_only"
        else:
            if epoch < warmup_epochs:
                effective_alpha = 0.0
                phase_name = "warmup"
            else:
                effective_alpha = float(alpha)
                phase_name = "finetune"
        
        # sampler의 epoch 설정
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
        # Training phase
        model.train()
        
        if is_main_process:
            train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        else:
            train_iter = train_loader
        
        loss_cls_sum_local = 0.0
        loss_rec_sum_local = 0.0
        loss_sum_local = 0.0   # sum of loss * batch_size
        correct_local  = 0.0
        total_local    = 0.0
        train_preds_local = []
        train_labels_local = []
        embeds_epoch, y_true_epoch, y_pred_epoch, ds_epoch = [], [], [], []
        embeds_diff_epoch = []
        reconstruction_image_to_log = None
        
        # dataset에서 getitem에 인자 true로 설정해놓으면 아래와 같이 info도 같이 줌
        for i, batch in enumerate(train_iter):
            if len(batch) == 3:
                inputs, labels, info = batch
                has_pair = False
                inputs_n = None
            else:
                inputs, labels, info, inputs_n, labels_n, info_n = batch     
                has_pair = True

            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if has_pair:
                inputs_n = inputs_n.to(device, non_blocking=True)

            optimizer.zero_grad()

            if config.reconstruct == "recon":
                reconstructed_img = net.reconstruct(inputs)
                loss_mae = nn.MSELoss()(reconstructed_img, inputs)
            elif config.reconstruct == "mae":
                reconstructed_img, _, masked_indices = net.forward_mae(img=inputs)
                loss_mae = net.calculate_mae_loss(reconstructed_img, inputs, masked_indices)
            cls_feat = net.get_features(inputs)

            predictions, diff_feat = net.forward_classify(current_img=inputs, normal_img=inputs_n)
            loss_classify = criterion(predictions, labels)

            loss = effective_alpha * loss_classify + (1 - effective_alpha) * loss_mae
        
            loss.backward()
            optimizer.step()

            # [수정] 메인 프로세스이고, 첫 번째 검증 배치일 때만 이미지 생성
            if is_main_process and i == 0 and wandb.run is not None:
                # 첫 번째 샘플에 대해 시각화 (inputs[0], rec_img[0])
                fig = create_reconstruction_figure(
                    orig_tensor=inputs[0],
                    rec_tensor=reconstructed_img[0],
                    mode=config.stft_mode,  # config에서 파라미터 가져오기
                    max_order=config.max_order,
                    window_sec=config.window_sec
                )
                # Figure를 wandb.Image 객체로 변환
                reconstruction_image_to_log = wandb.Image(fig)
                plt.close(fig)
            
            bs = labels.size(0)
            loss_cls_sum_local += loss_classify.item() * bs
            loss_rec_sum_local += loss_mae.item() * bs
            loss_sum_local += loss.item() * bs
            _, pred = predictions.max(1)
            correct_local += (pred == labels).sum().item()
            total_local += bs

            train_preds_local.append(pred.detach())
            train_labels_local.append(labels.detach())

            embeds_epoch.append(cls_feat.detach().cpu())     # (B, D)
            embeds_diff_epoch.append(diff_feat.detach().cpu())  # (B, D)
            y_true_epoch.append(labels.detach().cpu())
            y_pred_epoch.append(pred.detach().cpu())
            ds_epoch.extend(list(info["dataset"]))

            if is_main_process:
                avg_loss_so_far = loss_sum_local / max(total_local, 1)
                acc_so_far = 100.0 * correct_local / max(total_local, 1)
                train_iter.set_postfix({
                    'loss': f'{loss_sum_local/total_local:.4f}',
                    'acc': f'{100.*correct_local/total_local:.2f}%'
                })
        
        # reduce train metrics
        device0 = device

        train_preds_local = torch.cat(train_preds_local, dim=0)
        train_labels_local = torch.cat(train_labels_local, dim=0)
        tr_preds_np  = train_preds_local.detach().cpu().numpy()
        tr_labels_np = train_labels_local.detach().cpu().numpy()
        train_metrics = {}
        for avg in ["micro", "macro", "weighted"]:
            train_metrics[f"precision_{avg}"] = precision_score(tr_labels_np, tr_preds_np, average=avg, zero_division=0)
            train_metrics[f"recall_{avg}"] = recall_score(tr_labels_np, tr_preds_np, average=avg, zero_division=0)
            train_metrics[f"f1_{avg}"] = f1_score(tr_labels_np, tr_preds_np, average=avg, zero_division=0)
            
        t_train = torch.tensor([loss_sum_local, correct_local, total_local], dtype=torch.float64, device=device0)
        t_train_extra = torch.tensor([loss_cls_sum_local, loss_rec_sum_local], dtype=torch.float64, device=device0)
        train_loss = (t_train[0] / t_train[2]).item() if t_train[2] > 0 else 0.0
        train_acc = (t_train[1] / t_train[2] * 100.0).item() if t_train[2] > 0 else 0.0
        _ddp_sum_tensor(t_train_extra)
        _ddp_sum_tensor(t_train)
        train_loss_cls = (t_train_extra[0] / t_train[2]).item() if t_train[2] > 0 else 0.0
        train_loss_rec = (t_train_extra[1] / t_train[2]).item() if t_train[2] > 0 else 0.0

        # (3) 배치들 concat
        embeds_epoch = torch.cat(embeds_epoch, dim=0)    # (M, D)
        embeds_diff_epoch = torch.cat(embeds_diff_epoch, dim=0)    # (M, D)
        y_true_epoch = torch.cat(y_true_epoch, dim=0)    # (M,)
        y_pred_epoch = torch.cat(y_pred_epoch, dim=0)    # (M,)
        ds_epoch = np.array(ds_epoch)                    # (M,)
        
        project_cols = ("embedding", "pred", "label", "dataset", "split", "epoch")
        # (5) rank0만 로깅
        max_points = 4000
        M = embeds_epoch.shape[0]
        if M > max_points:
            idx = torch.randperm(M)[:max_points]
            embeds_s = embeds_epoch[idx]
            embeds_diff = embeds_diff_epoch[idx]
            ytrue_s = y_true_epoch[idx]
            ypred_s = y_pred_epoch[idx]
            # 리스트는 인덱싱으로 맞춰 재배치
            idx_np = idx.cpu().numpy().tolist()
            ds_s = [ds_epoch[i] for i in idx_np]
        else:
            embeds_s, embeds_diff, ytrue_s, ypred_s, ds_s = embeds_epoch, embeds_diff_epoch, y_true_epoch, y_pred_epoch, ds_epoch

        if rank == 0 and wandb.run is not None:
            # W&B Table (M rows)
            # 권장: Table 로깅 모드 지정 (INCREMENTAL/MUTABLE/IMMUTABLE) – 최근 가이드 참고
            # https://docs.wandb.ai/guides/models/tables/log_tables/
            if should_log_embed:
                table = wandb.Table(columns=list(project_cols), allow_mixed_types=True)

                E = embeds_s.cpu().numpy().tolist()
                P = ypred_s.cpu().numpy().tolist()
                Y = ytrue_s.cpu().numpy().tolist()

                for i in range(len(E)):
                    table.add_data(
                        E[i],
                        int(P[i]),
                        int(Y[i]),
                        ds_s[i],
                        "train",
                        int(epoch))
                # 한 번의 log 호출은 25MB 제한이 있으니(값 1MB 제한도 주의) 표본수를 조절
                # https://docs.wandb.ai/guides/track/limits/
                wandb.log({f"embeddings/train": table, "epoch": int(epoch)})
            
                table = wandb.Table(columns=list(project_cols), allow_mixed_types=True)

                E = embeds_diff.cpu().numpy().tolist()
                P = ypred_s.cpu().numpy().tolist()
                Y = ytrue_s.cpu().numpy().tolist()

                for i in range(len(E)):
                    table.add_data(
                        E[i],
                        int(P[i]),
                        int(Y[i]),
                        ds_s[i],
                        "train_diff",
                        int(epoch))
                # 한 번의 log 호출은 25MB 제한이 있으니(값 1MB 제한도 주의) 표본수를 조절
                # https://docs.wandb.ai/guides/track/limits/
                wandb.log({f"embeddings/train_diff": table, "epoch": int(epoch)})

        # Validation phase
        model.eval()
        val_loss_sum_local = 0.0
        val_loss_cls_sum_local = 0.0
        val_loss_rec_sum_local = 0.0
        val_correct_local  = 0.0
        val_total_local    = 0.0
        val_preds_local = []
        val_labels_local = []

        embeds_epoch, y_true_epoch, y_pred_epoch, ds_epoch = [], [], [], []
        embeds_diff_epoch = []
        val_reconstruction_image_to_log = None

        with torch.no_grad():
            for i, batch  in enumerate(val_loader):
                if len(batch) == 3:
                    inputs, labels, info = batch
                    has_pair = False
                    inputs_n = None
                else:
                    inputs, labels, info, inputs_n, labels_n, info_n = batch
                    has_pair = True
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if has_pair:
                    inputs_n = inputs_n.to(device, non_blocking=True)

                if config.reconstruct == "recon":
                    reconstructed_img = net.reconstruct(inputs)
                    b_loss_mae = nn.MSELoss()(reconstructed_img, inputs)
                elif config.reconstruct == "mae":
                    reconstructed_img, _, masked_indices = net.forward_mae(img=inputs)
                    b_loss_mae = net.calculate_mae_loss(reconstructed_img, inputs, masked_indices)
                cls_feat = net.get_features(inputs)

                predictions, diff_feat = net.forward_classify(current_img=inputs, normal_img=inputs_n)
                b_loss_classify = criterion(predictions, labels)

                b_loss = effective_alpha * b_loss_classify + (1 - effective_alpha) * b_loss_mae

                # [수정] 메인 프로세스이고, 첫 번째 검증 배치일 때만 이미지 생성
                if is_main_process and i == 0 and wandb.run is not None:
                    # 첫 번째 샘플에 대해 시각화 (inputs[0], rec_img[0])
                    fig = create_reconstruction_figure(
                        orig_tensor=inputs[0],
                        rec_tensor=reconstructed_img[0],
                        mode=config.stft_mode,  # config에서 파라미터 가져오기
                        max_order=config.max_order,
                        window_sec=config.window_sec
                    )
                    # Figure를 wandb.Image 객체로 변환
                    val_reconstruction_image_to_log = wandb.Image(fig)
                    plt.close(fig)

                bs = labels.size(0)
                val_loss_sum_local += b_loss.item() * bs
                val_loss_cls_sum_local += b_loss_classify.item() * bs
                val_loss_rec_sum_local += b_loss_mae.item() * bs
                _, pred = predictions.max(1)
                val_correct_local += (pred == labels).sum().item()
                val_total_local += bs
                val_preds_local.append(pred.detach())
                val_labels_local.append(labels.detach())
                
                embeds_epoch.append(cls_feat.detach().cpu())    # (B, D)
                embeds_diff_epoch.append(diff_feat.detach().cpu())    # (B, D)
                y_true_epoch.append(labels.detach().cpu())
                y_pred_epoch.append(pred.detach().cpu())
                ds_epoch.extend(list(info["dataset"]))
        
        val_preds_local = torch.cat(val_preds_local, dim=0)
        val_labels_local = torch.cat(val_labels_local, dim=0)
        val_preds_np = val_preds_local.detach().cpu().numpy()
        val_labels_np = val_labels_local.detach().cpu().numpy()
        val_metrics = {}
        for avg in ["micro", "macro", "weighted"]:
            val_metrics[f"precision_{avg}"] = precision_score(val_labels_np, val_preds_np, average=avg, zero_division=0)
            val_metrics[f"recall_{avg}"] = recall_score(val_labels_np, val_preds_np, average=avg, zero_division=0)
            val_metrics[f"f1_{avg}"] = f1_score(val_labels_np, val_preds_np, average=avg, zero_division=0)

        t_val = torch.tensor([val_loss_sum_local, val_correct_local, val_total_local], dtype=torch.float64, device=device0)
        t_val_extra = torch.tensor([val_loss_cls_sum_local, val_loss_rec_sum_local], dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_val_extra)
        _ddp_sum_tensor(t_val)
        val_loss_cls = (t_val_extra[0] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_loss_rec = (t_val_extra[1] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_loss = (t_val[0] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_acc  = (t_val[1] / t_val[2] * 100.0).item() if t_val[2] > 0 else 0.0


        # (3) 배치들 concat
        embeds_epoch = torch.cat(embeds_epoch, dim=0)    # (M, D)
        embeds_diff_epoch = torch.cat(embeds_diff_epoch, dim=0)    # (M, D)
        y_true_epoch = torch.cat(y_true_epoch, dim=0)    # (M,)
        y_pred_epoch = torch.cat(y_pred_epoch, dim=0)    # (M,)
        ds_epoch = np.array(ds_epoch)                    # (M,)
        
        project_cols = ("embedding", "pred", "label", "dataset", "split", "epoch")
        # (5) rank0만 로깅
        max_points = 4000
        M = embeds_epoch.shape[0]
        if M > max_points:
            idx = torch.randperm(M)[:max_points]
            embeds_s = embeds_epoch[idx]
            embeds_diff = embeds_diff_epoch[idx]
            ytrue_s = y_true_epoch[idx]
            ypred_s = y_pred_epoch[idx]
            # 리스트는 인덱싱으로 맞춰 재배치
            idx_np = idx.cpu().numpy().tolist()
            ds_s    = [ds_epoch[i] for i in idx_np]
        else:
            embeds_s, embeds_diff, ytrue_s, ypred_s, ds_s = embeds_epoch, embeds_diff_epoch, y_true_epoch, y_pred_epoch, ds_epoch

        if rank == 0 and wandb.run is not None:
            # W&B Table (M rows)
            # 권장: Table 로깅 모드 지정 (INCREMENTAL/MUTABLE/IMMUTABLE) – 최근 가이드 참고
            # https://docs.wandb.ai/guides/models/tables/log_tables/
            if should_log_embed:
                table = wandb.Table(columns=list(project_cols), allow_mixed_types=True)

                E = embeds_s.cpu().numpy().tolist()
                P = ypred_s.cpu().numpy().tolist()
                Y = ytrue_s.cpu().numpy().tolist()

                for i in range(len(E)):
                    table.add_data(
                        E[i],
                        int(P[i]),
                        int(Y[i]),
                        ds_s[i],
                        "val",
                        int(epoch))
                # 한 번의 log 호출은 25MB 제한이 있으니(값 1MB 제한도 주의) 표본수를 조절
                # https://docs.wandb.ai/guides/track/limits/
                wandb.log({f"embeddings/val": table, "epoch": int(epoch)})
                
                table = wandb.Table(columns=list(project_cols), allow_mixed_types=True)

                E = embeds_diff.cpu().numpy().tolist()
                P = ypred_s.cpu().numpy().tolist()
                Y = ytrue_s.cpu().numpy().tolist()

                for i in range(len(E)):
                    table.add_data(
                        E[i],
                        int(P[i]),
                        int(Y[i]),
                        ds_s[i],
                        "val_diff",
                        int(epoch))
                # 한 번의 log 호출은 25MB 제한이 있으니(값 1MB 제한도 주의) 표본수를 조절
                # https://docs.wandb.ai/guides/track/limits/
                wandb.log({f"embeddings/val_diff": table, "epoch": int(epoch)})


        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # Log metrics to wandb
        if is_main_process:  # 메인 프로세스에서만 로깅 및 모델 저장
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                "train/loss_cls": train_loss_cls,
                "train/loss_rec": train_loss_rec,
                "val/loss_cls": val_loss_cls,
                "val/loss_rec": val_loss_rec,
            }

            # 🌟 [수정] 생성된 이미지가 있으면 log_dict에 추가
            if reconstruction_image_to_log:
                log_dict['train/reconstruction_comparison'] = reconstruction_image_to_log
            if val_reconstruction_image_to_log:
                log_dict['val/reconstruction_comparison'] = val_reconstruction_image_to_log
            
            # 🔹 sklearn PRF 추가 (있을 때만)
            if train_metrics is not None:
                log_dict.update({
                    'train/precision_micro': train_metrics['precision_micro'],
                    'train/recall_micro': train_metrics['recall_micro'],
                    'train/f1_micro': train_metrics['f1_micro'],
                    'train/precision_macro': train_metrics['precision_macro'],
                    'train/recall_macro': train_metrics['recall_macro'],
                    'train/f1_macro': train_metrics['f1_macro'],
                    'train/precision_weighted': train_metrics['precision_weighted'],
                    'train/recall_weighted': train_metrics['recall_weighted'],
                    'train/f1_weighted': train_metrics['f1_weighted'],
                })
            if val_metrics is not None:
                log_dict.update({
                    'val/precision_micro': val_metrics['precision_micro'],
                    'val/recall_micro': val_metrics['recall_micro'],
                    'val/f1_micro': val_metrics['f1_micro'],
                    'val/precision_macro': val_metrics['precision_macro'],
                    'val/recall_macro': val_metrics['recall_macro'],
                    'val/f1_macro': val_metrics['f1_macro'],
                    'val/precision_weighted': val_metrics['precision_weighted'],
                    'val/recall_weighted': val_metrics['recall_weighted'],
                    'val/f1_weighted': val_metrics['f1_weighted'],
                })

            wandb.log(log_dict)

            

            # ====== Checkpointing ======
            # 1) last.pth (항상)
            last_path = os.path.join(save_dir, 'last.pth')
            save_checkpoint(last_path, model, optimizer, epoch,
                            meta={'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

            # 2) 주기 저장 (epoch_XXXX.pth)
            if save_every > 0 and ((epoch + 1) % save_every == 0):
                cyc_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.pth')
                save_checkpoint(cyc_path, model, optimizer, epoch,
                                meta={'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
                # 회전 삭제
                if max_keep > 0:
                    ckpts = sorted(glob.glob(os.path.join(save_dir, 'epoch_*.pth')))
                    if len(ckpts) > max_keep:
                        to_remove = ckpts[:len(ckpts) - max_keep]
                        for p in to_remove:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
            # 예: 재구성 관점이면 val_loss_rec, 전체는 val_loss
            if val_loss < best_val_loss:      # 변수명 유지 시 혼동 → best_val_loss 등으로 바꾸는 걸 권장
                best_val_loss = float(val_loss)  # .item() 보장
                os.makedirs('checkpoints', exist_ok=True)
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_val_loss
                }, os.path.join('checkpoints', 'best_model.pth'))
                wandb.save(os.path.join('checkpoints', 'best_model.pth'))
                print(f"[Epoch {epoch+1}/{num_epochs}] "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')



def setup(rank, world_size, args):
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    
    # CUDA 설정
    torch.cuda.set_device(rank)
    
    # 분산 처리 초기화
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    torch.distributed.destroy_process_group()

def build_normal_only_subset(ds: WindowedVibrationDataset, target_class: str = "normal"):
    is_normal_row = (ds.meta_df["merged_class"] == target_class).to_numpy()
    keep = [i for i, (row_idx, _start) in enumerate(ds.index_map) if is_normal_row[row_idx]]
    if not keep:
        raise RuntimeError(f"No '{target_class}' windows found in dataset.")
    return Subset(ds, keep)

def build_normal_only_subset_ddp(ds: WindowedVibrationDataset, target_class: str = "normal", rank: int = 0):
    if (not dist.is_available()) or (not dist.is_initialized()):
        return build_normal_only_subset(ds, target_class)

    if rank == 0:
        subset = build_normal_only_subset(ds, target_class)
        keep = subset.indices  # 원본 ds 기준 인덱스 리스트
    else:
        keep = None

    obj = [keep]
    dist.broadcast_object_list(obj, src=0)
    keep = obj[0]
    if keep is None or len(keep) == 0:
        raise RuntimeError("Broadcasted normal indices are empty.")
    return Subset(ds, keep)

def train_with_config(rank, world_size, args):
    
    setup(rank, world_size, args)

    # --- (1) 스윕 감지: 에이전트 실행 시 WANDB_SWEEP_ID 가 설정됨 ---
    is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))

    # wandb 초기화 (메인 프로세스에서만)
    if rank == 0:
        if is_sweep:
            run = wandb.init(project=args.project_name,
                             config=vars(args), 
                             name=args.run_name)
        else:
            # 일반 실행에서만 args를 config로 전달
            wandb.init(project=args.project_name,
                       config=vars(args), 
                       name=args.run_name)
    
    # --- (B) rank0의 config(dict) -> 모든 rank로 브로드캐스트 ---
    if rank == 0 and wandb.run is not None:
        # sweep일 경우 wandb.config가 최종값을 담고 있으므로 그것을 기준으로
        cfg_dict = dict(wandb.config)
    else:
        # 비-rank0는 임시로 args를 dict로
        cfg_dict = vars(args).copy()
    
    obj_list = [cfg_dict]
    dist.broadcast_object_list(obj_list, src=0)   # 모든 프로세스에 동일한 설정 전달
    cfg_dict = obj_list[0]                        # 동기화된 최종 설정

    # 모든 rank에서 공통으로 사용할 네임스페이스 구성
    config = types.SimpleNamespace(**cfg_dict)
    
    # 1) stft_pair가 있으면 "NxM" 형식으로 파싱
    if hasattr(config, "stft_pair") and config.stft_pair is not None:
        n_str, h_str = str(config.stft_pair).lower().split("x")
        config.stft_nperseg = int(n_str)
        config.stft_hop = int(h_str)
        if rank == 0 and wandb.run is not None:
            wandb.config.update({
                "stft_nperseg": config.stft_nperseg,
                "stft_hop": config.stft_hop
            }, allow_val_change=True)
    else:
        config.stft_nperseg = int(config.stft_nperseg)
        config.stft_hop = int(config.stft_hop)

    setattr(config, "stft_nperseg", config.stft_nperseg)
    setattr(config, "stft_hop", config.stft_hop)

    # W&B 설정 업데이트는 rank0에서만
    if rank == 0 and wandb.run is not None:
        wandb.config.update({
            "stft_nperseg": config.stft_nperseg,
            "stft_hop": config.stft_hop
        }, allow_val_change=True)

    torch.cuda.set_device(rank)  # 각 프로세스의 GPU 설정
    device = torch.device(f"cuda:{rank}")
    
    # 데이터 준비
    data_root = os.path.join(os.getcwd(), config.data_root)
    
    # 이미지 변환기 설정 (pretrained 모델 사용 시 224x224로 강제)
    output_size = 224 if config.pretrained else config.image_size
    if rank == 0 and config.pretrained and config.image_size != 224:
        print(f"Warning: Pretrained model requires 224x224 input. "
              f"Automatically adjusting output size from {config.image_size} to 224.")
    
    signal_imger = OrderInvariantSignalImager(
        mode=config.stft_mode,
        log1p=True,
        normalize="per_channel", 
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=config.max_order,
        H_out=output_size,
        W_out=output_size,
        stft_nperseg=config.stft_nperseg,
        stft_hop=config.stft_hop,
        stft_window="hann",
        stft_center=True,
        stft_power=config.stft_power,
    )
    
    # 학습용 데이터셋 생성
    train_dataset = WindowedVibrationDataset(
        data_root=data_root,
        using_dataset = ['dxai'],               # ['vat', 'vbl', 'mfd'],
        window_sec=config.window_sec,
        stride_sec=config.stride_sec,
        cache_mode='none',                      # file or none
        transform=signal_imger
    )

    # normal 분리 dataset
    normal_train_subset = build_normal_only_subset_ddp(train_dataset, target_class="normal", rank=rank)
    
    # 검증용 데이터셋 생성
    val_dataset = WindowedVibrationDataset(
        data_root=data_root,
        using_dataset = ['dxai'],
        window_sec=config.window_sec,
        stride_sec=config.stride_sec,
        cache_mode='none',                      # file or none
        transform=signal_imger
    )
    if config.recon_only is True:
        train_dataset = normal_train_subset
    # 분산 학습을 위한 sampler 생성
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size, 
                            sampler=train_sampler,
                            num_workers=4,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                          batch_size=config.batch_size, 
                          sampler=val_sampler,
                          num_workers=4,
                          pin_memory=True)
    
    # GPU 설정
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # 모델 생성 및 DDP 설정
    model = VisionTransformerAE(
        num_layers = 12,
        num_heads = 12,
        hidden_dim = 768,
        mlp_dim = 3072,
        dropout = 0.0,
        attention_dropout  = 0.0,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        image_size = 224,
        image_channel = 4,
        patch_size = 16,
        masking_ratio=0.75,
        num_classes=config.num_classes,
    ).to(device)

    if rank == 0:
        print(f"Creating model with {'pretrained' if config.pretrained else 'random'} initialization")
    
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    
    # config.resume_checkpoint가 제공된 경우, 상태 로드
    start_epoch = 0
    best_val_loss = float("inf")
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        # 각 프로세스에 맞는 GPU로 텐서를 매핑
        map_location = {'cuda:0': f'cuda:{rank}'}
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        unwrap_ddp(model).load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1
        if 'best_metric' in checkpoint:
            best_val_loss = float(checkpoint['best_metric'])
        if rank == 0:
            print(f"✅ Resumed from {config.resume_checkpoint} (start_epoch={start_epoch}, best={best_val_loss:.6f})")

    # 학습 실행
    train_model(
        alpha=config.alpha,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.epochs,
        device=device,
        start_epoch=start_epoch,
        rank=rank,
        config=config
    )
    
    cleanup()  # process group destroy
    if (rank == 0) and (wandb.run is not None):
        wandb.finish()

def run_training(args):
    # 사용 가능한 GPU 수 확인
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available")
    if world_size < 2:
        print("Warning: Less than 2 GPUs available. Using", world_size, "GPU(s)")
    
    # 메인 프로세스에서 필요한 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        mp.spawn(
            train_with_config,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    args = parse_args()
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU support.")
    
    # sweep 설정이 있는 경우
    if args.sweep_config:
        import yaml
        with open(args.sweep_config, 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_configuration, project=args.project_name)
        wandb.agent(sweep_id, function=lambda: run_training(args), count=50)
    else:
        # 일반 학습
        run_training(args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT for Vibration Diagnosis')
    parser.add_argument('--data_root', type=str, default='data/processed',
                        help='Path to the processed data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=0,
                        help='Save checkpoint every N epochs (0 = disable)')
    parser.add_argument('--max_keep', type=int, default=5,
                        help='Max number of periodic epoch_*.pth files to keep (FIFO). 0 = unlimited')
    parser.add_argument('--sweep_config', type=str, default=None,
                        help='Path to wandb sweep configuration file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--window_sec', type=float, default=5.0)
    parser.add_argument('--stride_sec', type=float, default=2.0)
    parser.add_argument('--max_order', type=float, default=20.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--stft_mode', type=str, default='stft+cross',
                        choices=['stft', 'stft+cross', 'stft_complex'])
    parser.add_argument('--stft_nperseg', type=int, default=1024,
                        help='Length of each STFT segment')
    parser.add_argument('--stft_hop', type=int, default=256,
                        help='Number of points between successive STFT segments')
    parser.add_argument('--stft_power', type=float, default=1.0,
                        help='Power of magnitude (1.0 for magnitude, 2.0 for power spectrum)')
    parser.add_argument('--project_name', type=str, default='vibration-diagnosis-recon')
    parser.add_argument('--recon_only', action='store_true',
                        help='Backprop on reconstruction loss only (no classifier loss)')
    parser.add_argument('--reconstruct', type=str, default='recon', choices=['recon', 'mae'])
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to the checkpoint file to resume training from.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='A custom name for the wandb run. If not set, wandb will generate one.')

    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Use ImageNet pretrained weights for ViT')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training')
    parser.add_argument('--warmup_epochs', type=int, default=1500,
                        help='epochs for reconstruction-only warm-up (classification weight=0)')


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU support.")
    
    # sweep 설정이 있는 경우
    if args.sweep_config:
        import yaml
        with open(args.sweep_config, 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_configuration, project=args.project_name)
        wandb.agent(sweep_id, function=lambda: run_training(args), count=5)
    else:
        # 일반 학습
        run_training(args)