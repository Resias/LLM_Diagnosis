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
from models.vit_encoder import VITEnClassify
from data.dataset import WindowedVibrationDataset, OrderInvariantSignalImager
import wandb
import ast
from tqdm import tqdm
import argparse


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
        
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            bs = labels.size(0)
            loss_sum_local += loss.item() * bs
            _, pred = outputs.max(1)
            correct_local += (pred == labels).sum().item()
            total_local += bs

            if is_main_process:
                avg_loss_so_far = loss_sum_local / max(total_local, 1)
                acc_so_far = 100.0 * correct_local / max(total_local, 1)
                train_iter.set_postfix({
                    'loss': f'{loss_sum_local/total_local:.4f}',
                    'acc': f'{100.*correct_local/total_local:.2f}%'
                })
        
        # reduce train metrics
        device0 = device
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
        
        t_val = torch.tensor([val_loss_sum_local, val_correct_local, val_total_local],
                             dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_val)
        val_loss = (t_val[0] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_acc  = (t_val[1] / t_val[2] * 100.0).item() if t_val[2] > 0 else 0.0

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # Log metrics to wandb
        if is_main_process:  # 메인 프로세스에서만 로깅 및 모델 저장
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

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
    os.environ['MASTER_PORT'] = '12355'
    
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

    # wandb 초기화 (메인 프로세스에서만)
    if rank == 0:
        if wandb.run is None:  # sweep이 아닌 경우에만 새로 초기화
            run = wandb.init(project=args.project_name, config=vars(args))
        config = wandb.config
    else:
        config = args  # 다른 프로세스는 args를 직접 사용
        
    torch.cuda.set_device(rank)  # 각 프로세스의 GPU 설정
    device = torch.device(f"cuda:{rank}")
    
    # 데이터 준비
    data_root = os.path.join(os.getcwd(), config.data_root)
    meta_csv = os.path.join(data_root, 'meta.csv')
    meta_pd = pd.read_csv(meta_csv)
    
    meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
    meta_pd = meta_pd[5 <= meta_pd['data_sec']]
    
    using_dataset = ['dxai', 'vat', 'vbl', 'mfd']
    meta_pd = meta_pd[meta_pd['dataset'].isin(using_dataset)]
    
    # 이미지 변환기 설정
    signal_imger = OrderInvariantSignalImager(
        mode=config.stft_mode,
        log1p=True,
        normalize="per_channel",
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=config.max_order,
        H_out=config.image_size,
        W_out=config.image_size,
        stft_nperseg=config.stft_nperseg,
        stft_hop=config.stft_hop,
        stft_window="hann",
        stft_center=True,
        stft_power=config.stft_power,
        # CWT
        cwt_wavelet="morl",
        cwt_num_scales=64,
        cwt_scale_base=2.0,
    )
    
    # 데이터셋 생성
    dataset = WindowedVibrationDataset(
        meta_df=meta_pd,
        data_root=data_root,
        window_sec=config.window_sec,
        stride_sec=config.stride_sec,
        cache_mode='file',
        transform=signal_imger
    )
    
    # 데이터셋 분할 (8:2)
    generator = torch.Generator().manual_seed(42)  # 재현성을 위한 시드 설정
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
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
    model = VITEnClassify(
        num_classes=config.num_classes,
        image_size=config.image_size,
        patch_size=16
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # wandb에 모델 구조 기록 (메인 프로세스만)
    if rank == 0:
        wandb.watch(model, criterion, log="all", log_freq=1000)
    
    # 학습 실행
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.epochs,
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
        wandb.agent(sweep_id, function=lambda: run_training(args), count=5)
    else:
        # 일반 학습
        run_training(args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT for Vibration Diagnosis')
    parser.add_argument('--data_root', type=str, default='data/processed',
                        help='Path to the processed data directory')
    parser.add_argument('--sweep_config', type=str, default=None,
                        help='Path to wandb sweep configuration file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--window_sec', type=float, default=5.0)
    parser.add_argument('--stride_sec', type=float, default=2.0)
    parser.add_argument('--max_order', type=float, default=20.0)
    parser.add_argument('--stft_mode', type=str, default='stft+cross',
                        choices=['stft', 'stft+cross', 'stft_complex'])
    parser.add_argument('--stft_nperseg', type=int, default=1024,
                        help='Length of each STFT segment')
    parser.add_argument('--stft_hop', type=int, default=256,
                        help='Number of points between successive STFT segments')
    parser.add_argument('--stft_power', type=float, default=1.0,
                        help='Power of magnitude (1.0 for magnitude, 2.0 for power spectrum)')
    parser.add_argument('--project_name', type=str, default='vibration-diagnosis')
    
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