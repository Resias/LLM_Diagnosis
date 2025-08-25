import os
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from models.resnet_encoder import ResNetEnClassify
from data.dataset import WindowedVibrationDataset, OrderInvariantSignalImager
from sklearn.metrics import precision_score, recall_score, f1_score

import wandb
import ast
from tqdm import tqdm
import argparse
import types



def _ddp_sum_tensor(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, rank):
    best_val_acc = 0.0
    is_main_process = rank == 0  # 메인 프로세스 여부 확인
    
    for epoch in range(num_epochs):
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
        loss_sum_local = 0.0   # sum of loss * batch_size
        correct_local  = 0.0
        total_local    = 0.0
        train_preds_local = []
        train_labels_local = []
        
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # print(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            
            bs = labels.size(0)
            loss_sum_local += loss.item() * bs
            _, pred = outputs.max(1)
            correct_local += (pred == labels).sum().item()
            total_local += bs

            train_preds_local.append(pred.detach())
            train_labels_local.append(labels.detach())

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
            
        t_train = torch.tensor([loss_sum_local, correct_local, total_local],
                               dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_train)
        train_loss = (t_train[0] / t_train[2]).item() if t_train[2] > 0 else 0.0
        train_acc  = (t_train[1] / t_train[2] * 100.0).item() if t_train[2] > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss_sum_local = 0.0
        val_correct_local  = 0.0
        val_total_local    = 0.0
        val_preds_local = []
        val_labels_local = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                bs = labels.size(0)
                val_loss_sum_local += loss.item() * bs
                _, pred = outputs.max(1)
                val_correct_local += (pred == labels).sum().item()
                val_total_local += bs
                val_preds_local.append(pred.detach())
                val_labels_local.append(labels.detach())
        
        val_preds_local = torch.cat(val_preds_local, dim=0)
        val_labels_local = torch.cat(val_labels_local, dim=0)
        val_preds_np  = val_preds_local.detach().cpu().numpy()
        val_labels_np = val_labels_local.detach().cpu().numpy()
        val_metrics = {}
        for avg in ["micro", "macro", "weighted"]:
            val_metrics[f"precision_{avg}"] = precision_score(val_labels_np, val_preds_np, average=avg, zero_division=0)
            val_metrics[f"recall_{avg}"] = recall_score(val_labels_np, val_preds_np, average=avg, zero_division=0)
            val_metrics[f"f1_{avg}"] = f1_score(val_labels_np, val_preds_np, average=avg, zero_division=0)

        t_val = torch.tensor([val_loss_sum_local, val_correct_local, val_total_local],
                             dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_val)
        val_loss = (t_val[0] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_acc  = (t_val[1] / t_val[2] * 100.0).item() if t_val[2] > 0 else 0.0

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # Log metrics to wandb
        if is_main_process:  # 메인 프로세스에서만 로깅 및 모델 저장
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
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

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('checkpoints', exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # DDP unwrap
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
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

def train_with_config(rank, world_size, args):
    setup(rank, world_size, args)
    is_main = (rank == 0)
    
    # 스윕 여부 판단 (wandb agent가 설정하는 환경변수)
    is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    
    # rank0만 wandb.init (스윕이면 config를 절대 넘기지 않음)
    if is_main:
        if is_sweep:
            wandb.init(project=args.project_name)
        else:
            wandb.init(project=args.project_name, config=vars(args))

        # 1) rank0에서 최종 설정 만들기: args 기본값으로 시작 → 스윕값 덮어쓰기
        merged = vars(args).copy()
        if wandb.run is not None:
            merged.update(dict(wandb.config))  # 스윕 값이 우선

        # 2) stft_pair 정규화 (있으면 stft_nperseg/hop 생성·덮어쓰기)
        stft_pair = merged.get("stft_pair", None)
        if stft_pair is not None:
            pair_str = str(stft_pair).lower()
            try:
                n_str, h_str = pair_str.split("x")
                merged["stft_nperseg"] = int(n_str)
                merged["stft_hop"]     = int(h_str)
            except Exception as e:
                raise ValueError(f"Invalid stft_pair format: {stft_pair} (expected 'NxM')") from e
        else:
            if "stft_nperseg" not in merged or "stft_hop" not in merged:
                raise ValueError("Provide either stft_pair or both stft_nperseg and stft_hop.")

        # (선택) wandb.config에 최종 하이퍼 파라미터 기록(로깅용)
        if wandb.run is not None:
            wandb.config.update({
                "epochs": int(merged["epochs"]),
                "batch_size": int(merged["batch_size"]),
                "learning_rate": float(merged["learning_rate"]),
                "stft_nperseg": int(merged["stft_nperseg"]),
                "stft_hop": int(merged["stft_hop"]),
                "model_select": int(merged.get("model_select", merged.get("model-select", 18))),
                "pretrained": bool(merged.get("pretrained", False)),
                "image_size": int(merged["image_size"]),
            }, allow_val_change=True)

        # (디버그) 최종 설정 확인
        print("[CONFIG/rank0]", merged)

    else:
        merged = vars(args).copy()
    
    # ---- 모든 rank로 브로드캐스트 ----
    obj_list = [merged]
    dist.broadcast_object_list(obj_list, src=0)
    merged = obj_list[0]
    
    # argparse 네임 규칙: --model-select → model_select 로 정규화
    if "model_select" not in merged and "model-select" in merged:
        merged["model_select"] = merged["model-select"]

    # 중요: 이후 전부 args만 사용하도록 덮어쓰기
    args = types.SimpleNamespace(**merged)

    torch.cuda.set_device(rank)  # 각 프로세스의 GPU 설정
    device = torch.device(f"cuda:{rank}")
    
    # 데이터 준비
    data_root = os.path.join(os.getcwd(), args.data_root)
    meta_csv = os.path.join(data_root, 'meta.csv')
    meta_pd = pd.read_csv(meta_csv)
    
    meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
    meta_pd = meta_pd[5 <= meta_pd['data_sec']]
    
    # 학습용 데이터셋과 검증용 데이터셋 분리
    train_meta = meta_pd[meta_pd['dataset'] != 'dxai'].copy()
    val_meta = meta_pd[meta_pd['dataset'] == 'dxai'].copy()
    
    if is_main:
        print(f"\nTraining samples (non dxai): {len(train_meta)}")
        print(f"Validation samples (dxai): {len(val_meta)}")
        print("\nTraining dataset distribution:")
        print(train_meta['dataset'].value_counts())
        print("\nValidation dataset distribution:")
        print(val_meta['dataset'].value_counts())
    
    # 이미지 변환기 설정 (pretrained 모델 사용 시 224x224로 강제)
    output_size = 224 if bool(args.pretrained) else int(args.image_size)
    if is_main and bool(args.pretrained) and int(args.image_size) != 224:
        print(f"Warning: Pretrained model requires 224x224 input. "
              f"Automatically adjusting output size from {args.image_size} to 224.")
    
    signal_imger = OrderInvariantSignalImager(
        mode=args.stft_mode,
        log1p=True,
        normalize="none",
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=args.max_order,
        H_out=output_size,
        W_out=output_size,
        stft_nperseg=args.stft_nperseg,
        stft_hop=args.stft_hop,
        stft_window="hann",
        stft_center=True,
        stft_power=args.stft_power,
        # CWT
        cwt_wavelet="morl",
        cwt_num_scales=64,
        cwt_scale_base=2.0,
    )
    
    # 학습용 데이터셋 생성
    train_dataset = WindowedVibrationDataset(
        meta_df=train_meta,
        data_root=data_root,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        cache_mode='none',
        transform=signal_imger
    )
    
    # 검증용 데이터셋 생성
    val_dataset = WindowedVibrationDataset(
        meta_df=val_meta,
        data_root=data_root,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        cache_mode='none',
        transform=signal_imger
    )
    
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
                            batch_size=args.batch_size, 
                            sampler=train_sampler,
                            num_workers=4,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                          batch_size=args.batch_size, 
                          sampler=val_sampler,
                          num_workers=4,
                          pin_memory=True)
    
    # GPU 설정
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # 모델 생성 및 DDP 설정
    model = ResNetEnClassify(
        res_select=args.model_select,
        num_classes=args.num_classes,
        image_size=output_size,
        pretrained=args.pretrained
    ).to(device)

    if is_main:
        print(f"Creating model with {'pretrained' if args.pretrained else 'random'} initialization")

    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = int(args.epochs)

    # 학습 실행
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        rank=rank
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
    parser.add_argument('--sweep_config', type=str, default=None,
                        help='Path to wandb sweep configuration file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--window_sec', type=float, default=5.0)
    parser.add_argument('--stride_sec', type=float, default=2.0)
    parser.add_argument('--max_order', type=float, default=10.0)
    parser.add_argument('--stft_mode', type=str, default='stft+cross',
                        choices=['stft', 'stft+cross', 'stft_complex'])
    parser.add_argument('--stft_nperseg', type=int, default=1024,
                        help='Length of each STFT segment')
    parser.add_argument('--stft_hop', type=int, default=256,
                        help='Number of points between successive STFT segments')
    parser.add_argument('--stft_power', type=float, default=1.0,
                        help='Power of magnitude (1.0 for magnitude, 2.0 for power spectrum)')
    parser.add_argument('--project_name', type=str, default='vibration-diagnosis-ood')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Use ImageNet pretrained weights for ViT')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training')
    parser.add_argument('--model-select', type=int, default=18,
                        help='Select ResNet model (18 or 50)')

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