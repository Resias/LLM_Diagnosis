from datetime import datetime
import os
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from functools import partial

from data.dataset import VibrationDataset, OrderInvariantSignalImager
from sklearn.metrics import precision_score, recall_score, f1_score

from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE

import wandb
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import types

def unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model

def _ddp_sum_tensor(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def _to_uint8_image(t: torch.Tensor):
    """
    t: (C,H,W) float/half, arbitrary range. -> uint8 HxW or HxW xC
    1) detach->cpu  2) min-max normalize per-image  3) to uint8
    """
    t = t.detach().float().cpu()
    if t.dim() == 3:  # C,H,W
        # 시각화 채널 결정: C>=3이면 처음 3채널, 그 외엔 채널 평균
        if t.size(0) >= 3:
            img = t[:3]  # (3,H,W)
        else:
            img = t.mean(0, keepdim=True).repeat(3,1,1)  # (3,H,W)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255.0).clamp(0,255).byte().permute(1,2,0).numpy()  # H,W,3
        return img
    elif t.dim() == 2:  # H,W
        img = (t - t.min()) / (t.max() - t.min() + 1e-8)
        img = (img * 255.0).clamp(0,255).byte().numpy()
        return img
    else:
        raise ValueError("Unexpected tensor shape for image.")

def log_recon_image_wandb(orig: torch.Tensor, rec: torch.Tensor, tag: str, epoch: int, downsample: int = 2):
    """
    orig/rec: (C,H,W). 메모리 절약을 위해 다운샘플, uint8 변환 후 wandb.Image로 로깅.
    """
    # 다운샘플 (stride indexing; 보간 없음 → 빠름)
    if downsample > 1:
        orig = orig[:, ::downsample, ::downsample]
        rec  = rec[:,  ::downsample, ::downsample]
    orig_img = _to_uint8_image(orig)
    rec_img  = _to_uint8_image(rec)

    # 간단 비교 figure (2열)
    fig, axes = plt.subplots(1, 2, figsize=(6,3), dpi=100)
    axes[0].imshow(orig_img); axes[0].axis("off"); axes[0].set_title("orig")
    axes[1].imshow(rec_img);  axes[1].axis("off"); axes[1].set_title("recon")
    wandb.log({tag: wandb.Image(fig), "epoch": int(epoch)})
    plt.close(fig)

def sample_rows_for_table(arr_list, max_rows: int):
    """
    arr_list 길이가 max_rows보다 크면 균일 샘플. 작은 경우 그대로 반환.
    """
    n = len(arr_list)
    if n <= max_rows:
        return arr_list
    step = n / max_rows
    idx = [int(i*step) for i in range(max_rows)]
    return [arr_list[i] for i in idx]

def log_embeddings_table_wandb(emb: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
                               split: str, epoch: int, max_rows: int = 200, tag_prefix: str = "emb"):
    """
    emb: (B,D) on GPU
    y_true, y_pred: (B,)
    → CPU로 옮겨 소량만 테이블에 기록. 거대한 테이블/누적 금지.
    """
    if emb is None or emb.numel() == 0:
        return
    with torch.no_grad():
        E = emb.detach().float().cpu().numpy().tolist()
        Y = y_true.detach().cpu().tolist()
        P = y_pred.detach().cpu().tolist()

    # 행 샘플링
    rows = list(zip(E, P, Y))
    rows = sample_rows_for_table(rows, max_rows)

    table = wandb.Table(columns=["embedding", "pred", "label"], allow_mixed_types=True)
    for e, p, y in rows:
        table.add_data(e, int(p), int(y))
    wandb.log({f"{tag_prefix}/{split}": table, "epoch": int(epoch)})

def ddp_gather_small_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    각 rank의 (K, D) 텐서를 all_gather로 모아 rank0에서 cat.
    K가 작을 때만 사용(임베딩 일부 샘플). 크면 비용 ↑
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world_size = dist.get_world_size()
    tensors_gather = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(tensors_gather, t)
    return torch.cat(tensors_gather, dim=0)

def vib_collate(batch):
    out = {}
    must_stack = {"x_stft", "ref_stft", "x_cls"}  # 모델이 바로 쓰는 키만 stack
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if k in must_stack:
            # tensor로 변환 보장
            vals = [torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in vals]
            out[k] = torch.stack(vals, dim=0)
        else:
            # 길이 제각각일 수 있으니 리스트로 유지
            out[k] = vals
    return out

def train_model(alpha, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, rank, config):
    net = unwrap_ddp(model)
    is_main = (rank == 0)
    best_val_acc = 0.0
    warmup_epochs = int(getattr(config, "warmup_epochs", 0))

    # 성능 옵션
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    # AMP
    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # 로깅 주기/용량 제한
    IMG_LOG_EVERY = getattr(config, "img_log_every", 100)
    EMB_LOG_EVERY = getattr(config, "emb_log_every", 100)
    EMB_PER_RANK  = getattr(config, "emb_per_rank", 32)
    TABLE_MAX_ROWS = getattr(config, "table_max_rows", 200)
    
    for epoch in range(num_epochs):
        effective_alpha = 0.0 if epoch < warmup_epochs else float(alpha)

        # sampler epoch advance
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(epoch)

        # ---------------- Train ----------------
        # Training phase
        model.train()
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if is_main else train_loader

        loss_sum_local = 0.0
        correct_local = 0.0
        total_local = 0.0
        train_preds_local = []
        train_labels_local = []

        # 마지막 배치의 diff/logits/y (임베딩 로깅용)만 잡아둠. 누적 금지.
        last_train_diff = None
        last_train_logits = None
        last_train_y = None
        first_train_image_logged = False
        
        # dataset에서 getitem에 인자 true로 설정해놓으면 아래와 같이 info도 같이 줌
        for i, batch in enumerate(train_iter):
            x = batch['x_stft'].to(device, non_blocking=True)
            y = batch['x_cls'].to(device, non_blocking=True)
            ref_x = batch['ref_stft'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                rec, _, masked_idx = net.forward_mae(img=x)
                # loss_mae = net.calculate_mae_loss(rec, x, masked_idx)
                loss_mae = nn.MSELoss()(rec, x)

                logits, diff = net.forward_classify(current_img=x, normal_img=ref_x)
                loss_cls = criterion(logits, y)

                loss = effective_alpha * loss_cls + (1.0 - effective_alpha) * loss_mae

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            loss_sum_local += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct_local += (pred == y).sum().item()
            total_local += bs

            train_preds_local.append(pred.detach())
            train_labels_local.append(y.detach())
            
            # 이미지 로깅: 첫 배치에서만, 큰 간격으로
            if (not first_train_image_logged) and is_main and wandb.run is not None and (epoch % IMG_LOG_EVERY == 0):
                # 배치의 첫 샘플만 사용
                try:
                    log_recon_image_wandb(x[0], rec[0], tag="images/train_recon", epoch=epoch, downsample=2)
                    first_train_image_logged = True
                except Exception:
                    pass  # 로깅 실패해도 학습은 계속

            # 임베딩 테이블은 에폭 말에 한 번만 기록할 것이므로 마지막 배치 참조만 유지
            last_train_diff = diff.detach()
            last_train_logits = logits.detach()
            last_train_y = y.detach()

            if is_main and isinstance(train_iter, tqdm):
                train_iter.set_postfix({
                    "loss": f"{loss_sum_local/max(total_local,1):.4f}",
                    "acc":  f"{100.0*correct_local/max(total_local,1):.2f}%"
                })
        
        # reduce train metrics
        device0 = device

        t_train = torch.tensor([loss_sum_local, correct_local, total_local], dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_train)
        train_loss = (t_train[0] / t_train[2]).item() if t_train[2] > 0 else 0.0
        train_acc = (t_train[1] / t_train[2] * 100.0).item() if t_train[2] > 0 else 0.0

        # sklearn metrics (rank별 로컬; 보고용)
        train_preds_local = torch.cat(train_preds_local, dim=0).detach().cpu().numpy()
        train_labels_local = torch.cat(train_labels_local, dim=0).detach().cpu().numpy()
        train_metrics = {
            "precision_micro":  precision_score(train_labels_local, train_preds_local, average="micro", zero_division=0),
            "recall_micro":     recall_score(train_labels_local, train_preds_local, average="micro", zero_division=0),
            "f1_micro":         f1_score(train_labels_local, train_preds_local, average="micro", zero_division=0),
            "precision_macro":  precision_score(train_labels_local, train_preds_local, average="macro", zero_division=0),
            "recall_macro":     recall_score(train_labels_local, train_preds_local, average="macro", zero_division=0),
            "f1_macro":         f1_score(train_labels_local, train_preds_local, average="macro", zero_division=0),
            "precision_weighted": precision_score(train_labels_local, train_preds_local, average="weighted", zero_division=0),
            "recall_weighted":    recall_score(train_labels_local, train_preds_local, average="weighted", zero_division=0),
            "f1_weighted":        f1_score(train_labels_local, train_preds_local, average="weighted", zero_division=0),
        }
        
        # ---------------- Val ----------------
        # Validation phase
        model.eval()
        val_loss_sum_local = 0.0
        val_correct_local  = 0.0
        val_total_local    = 0.0
        val_preds_local = []
        val_labels_local = []

        last_val_diff = None
        last_val_logits = None
        last_val_y = None
        first_val_image_logged = False

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if is_main else val_loader

        with torch.no_grad():
            for i, batch  in enumerate(val_iter):
                x = batch['x_stft'].to(device, non_blocking=True)
                y = batch['x_cls'].to(device, non_blocking=True)
                ref_x = batch['ref_stft'].to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    rec, _, masked_idx = net.forward_mae(img=x)
                    # b_loss_mae = net.calculate_mae_loss(rec, x, masked_idx)
                    b_loss_mae = nn.MSELoss()(rec, x)

                    logits, diff = net.forward_classify(current_img=x, normal_img=ref_x)
                    b_loss_cls = criterion(logits, y)

                    b_loss = effective_alpha * b_loss_cls + (1.0 - effective_alpha) * b_loss_mae

                bs = y.size(0)
                val_loss_sum_local += b_loss.item() * bs
                pred = logits.argmax(dim=1)
                val_correct_local += (pred == y).sum().item()
                val_total_local += bs

                val_preds_local.append(pred.detach())
                val_labels_local.append(y.detach())

                # 이미지 로깅: 첫 배치에서만, 큰 간격으로
                if (not first_val_image_logged) and is_main and wandb.run is not None and (epoch % IMG_LOG_EVERY == 0):
                    try:
                        log_recon_image_wandb(x[0], rec[0], tag="images/val_recon", epoch=epoch, downsample=2)
                        first_val_image_logged = True
                    except Exception:
                        pass

                # 임베딩 로깅용 참조 저장(마지막 배치)
                last_val_diff = diff.detach()
                last_val_logits = logits.detach()
                last_val_y = y.detach()
        
        t_val = torch.tensor([val_loss_sum_local, val_correct_local, val_total_local], dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_val)
        val_loss = (t_val[0] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_acc  = (t_val[1] / t_val[2] * 100.0).item() if t_val[2] > 0 else 0.0

        val_preds_local = torch.cat(val_preds_local, dim=0).detach().cpu().numpy()
        val_labels_local = torch.cat(val_labels_local, dim=0).detach().cpu().numpy()
        val_metrics = {
            "precision_micro":  precision_score(val_labels_local, val_preds_local, average="micro", zero_division=0),
            "recall_micro":     recall_score(val_labels_local, val_preds_local, average="micro", zero_division=0),
            "f1_micro":         f1_score(val_labels_local, val_preds_local, average="micro", zero_division=0),
            "precision_macro":  precision_score(val_labels_local, val_preds_local, average="macro", zero_division=0),
            "recall_macro":     recall_score(val_labels_local, val_preds_local, average="macro", zero_division=0),
            "f1_macro":         f1_score(val_labels_local, val_preds_local, average="macro", zero_division=0),
            "precision_weighted": precision_score(val_labels_local, val_preds_local, average="weighted", zero_division=0),
            "recall_weighted":    recall_score(val_labels_local, val_preds_local, average="weighted", zero_division=0),
            "f1_weighted":        f1_score(val_labels_local, val_preds_local, average="weighted", zero_division=0),
        }

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # Log metrics to wandb
        
        # ---------------- Light Logging (scalars) ----------------
        if is_main:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "train/precision_micro": train_metrics["precision_micro"],
                "train/recall_micro":    train_metrics["recall_micro"],
                "train/f1_micro":        train_metrics["f1_micro"],
                "train/precision_macro": train_metrics["precision_macro"],
                "train/recall_macro":    train_metrics["recall_macro"],
                "train/f1_macro":        train_metrics["f1_macro"],
                "train/precision_weighted": train_metrics["precision_weighted"],
                "train/recall_weighted":    train_metrics["recall_weighted"],
                "train/f1_weighted":        train_metrics["f1_weighted"],
                "val/precision_micro":  val_metrics["precision_micro"],
                "val/recall_micro":     val_metrics["recall_micro"],
                "val/f1_micro":         val_metrics["f1_micro"],
                "val/precision_macro":  val_metrics["precision_macro"],
                "val/recall_macro":     val_metrics["recall_macro"],
                "val/f1_macro":         val_metrics["f1_macro"],
                "val/precision_weighted": val_metrics["precision_weighted"],
                "val/recall_weighted":    val_metrics["recall_weighted"],
                "val/f1_weighted":        val_metrics["f1_weighted"],
            }
            if wandb.run is not None:
                wandb.log(log_dict)

            # 베스트 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('checkpoints', exist_ok=True)
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join('checkpoints', 'best_model.pth'))

            print(f"[{epoch+1}/{num_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        # ---------------- Light Logging (embeddings table) ----------------
        # 에폭 주기마다, 배치 '마지막' diff에서만 소량 샘플링해서 테이블에 올림
        if wandb.run is not None and (epoch % EMB_LOG_EVERY == 0):
            # Train
            if last_train_diff is not None and last_train_logits is not None and last_train_y is not None:
                K = min(EMB_PER_RANK, last_train_diff.size(0))
                idx = torch.randint(0, last_train_diff.size(0), (K,), device=last_train_diff.device)
                emb_small  = last_train_diff[idx]
                y_small    = last_train_y[idx]
                pred_small = last_train_logits.argmax(dim=1)[idx]

                # 여러 rank 샘플 모으기(작을 때만)
                emb_small  = ddp_gather_small_tensor(emb_small)
                y_small    = ddp_gather_small_tensor(y_small)
                pred_small = ddp_gather_small_tensor(pred_small)

                if is_main:
                    log_embeddings_table_wandb(
                        emb_small, y_small, pred_small,
                        split="train_diff", epoch=epoch,
                        max_rows=TABLE_MAX_ROWS, tag_prefix="emb"
                    )

            # Val
            if last_val_diff is not None and last_val_logits is not None and last_val_y is not None:
                K = min(EMB_PER_RANK, last_val_diff.size(0))
                idx = torch.randint(0, last_val_diff.size(0), (K,), device=last_val_diff.device)
                emb_small  = last_val_diff[idx]
                y_small    = last_val_y[idx]
                pred_small = last_val_logits.argmax(dim=1)[idx]

                emb_small  = ddp_gather_small_tensor(emb_small)
                y_small    = ddp_gather_small_tensor(y_small)
                pred_small = ddp_gather_small_tensor(pred_small)

                if is_main:
                    log_embeddings_table_wandb(
                        emb_small, y_small, pred_small,
                        split="val_diff", epoch=epoch,
                        max_rows=TABLE_MAX_ROWS, tag_prefix="emb"
                    )



def setup(rank, world_size, args):
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    
    # 분산 처리 초기화
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    # CUDA 설정
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_with_config(rank, world_size, args):
    setup(rank, world_size, args)

    # wandb 초기화 (메인 프로세스에서만)
    if rank == 0:
        wandb.init(project=args.project_name, config=vars(args))
    
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
    train_dataset = VibrationDataset(
        data_root=data_root,
        using_dataset = ['vat', 'vbl', 'mfd', 'dxai'],
        window_sec=config.window_sec,
        stride_sec=config.stride_sec,
        transform=signal_imger
    )
    
    # 검증용 데이터셋 생성
    val_dataset = VibrationDataset(
        data_root=data_root,
        using_dataset = ['dxai'],
        window_sec=config.window_sec,
        stride_sec=config.stride_sec,
        transform=signal_imger
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # 워커 수는 보수적으로
    workers = max(2, min(4, (os.cpu_count() or 8) // max(1, 2*world_size)))
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size, 
                            num_workers=2,
                            sampler=train_sampler,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=vib_collate)
    val_loader = DataLoader(val_dataset, 
                          batch_size=config.batch_size, 
                          num_workers=2,
                          sampler=val_sampler,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=vib_collate)
    
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
        # mp.spawn(
        #     train_with_config,
        #     args=(world_size, args),
        #     nprocs=world_size,
        #     join=True
        # )
        train_with_config(rank=0, world_size=world_size, args=args)
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--window_sec', type=float, default=5.0)
    parser.add_argument('--stride_sec', type=float, default=3.0)
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
    parser.add_argument('--project_name', type=str, default='vibration-diagnosis-final_1105night_2050')
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
    
    run_training(args)