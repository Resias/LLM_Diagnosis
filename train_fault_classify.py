import torch
from torch.utils.data import random_split, DataLoader

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler


from models.unet import UnetVAE
from models.classifier import ClassifierUnet
from models.trainer import Trainer
from data.dataset import VibrationDataset, CachedDataset, count_classes

import os
import wandb
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Train Fault classify model Config")
    parser.add_argument('--project_name', type=str, help='wandb project name for trainining', default='phm_LLM')
    parser.add_argument('--run_name', type=str, help='wandb run name for fixed', default=None)

    # 데이터셋 관련 옵션들
    parser.add_argument('--dataset_root', type=str, help='dataset_root', default='/home/data/')
    parser.add_argument('--channels_used', type=list, help='select using channels for training', default=['motor_x', 'motor_y'])
    parser.add_argument('--dataset_used', type=list, help='dataset for training', default = ['dxai', 'iis', 'mfd', 'vat', 'vbl'])
    parser.add_argument('--class_used', type=list, help='classes for training', default=['looseness', 'normal', 'unbalance','misalignment', 'bearing'])

    parser.add_argument('--use_cache', type=int, help='use cache dataset', choices=[0, 1], default=1)
    parser.add_argument('--cache_name', type=str, help='cached dataset Name, format trainset_pipeline_\{preprocessing_type\}_\{cache_name\}', default=None)    
    
    # 모델 관련 옵션
    parser.add_argument('--embedding_dim', type=int, help='Encoder/Decoder final shape : embedding_dim x num_embeddings', default=32)
    parser.add_argument('--num_embeddings', type=int, help='after/before Encoder/Decoder shape :embeddings x num_embeddings', default=128)
    parser.add_argument('--num_residual_layers', type=int, help='number of residual layers for each residual blocks', default=8)
    parser.add_argument('--num_residual_hiddens', type=int, help='feature expension in residual connection : 2 means 2 times', default=2)

    # 학습 관련 옵션
    parser.add_argument('--batch_size', type=int, help='batch_size', default=512)
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs', default=400)
    parser.add_argument('--recon_loss', type=str, help='reconstruction loss type for training', default='huber')
    parser.add_argument('--class_loss', type=str, help='classifier loss type for training', default='focal')
    parser.add_argument('--only_encoder', type=int, help='use only encoder and no reconsturcion', choices=[0, 1], default=0)
    parser.add_argument('--only_reconstruction', type=int, help='use only reconsturcion', choices=[0, 1], default=0)

    # Test Time Training argments
    parser.add_argument('--apply_ttt', type=int, help='applying ttt', choices=[0, 1], default=0)    
    parser.add_argument('--test_epochs', type=int, help='Maximum test epochs', default=10)
    parser.add_argument('--test_batch_size', type=int, help='test batch size', default=512)
    parser.add_argument('--ttt_step', type=int, help='TTT steps', default=5)
    parser.add_argument('--ttt_lr', type=float, help='TTT learning rate under 1e-5', default=1e-5)    
    parser.add_argument('--anormaly_threshold', type=float, help='anormaly threshold (0.0 ~ 1.0)', default=0.5)

    parser.add_argument('--fast_dev_run', type=bool, help='pytorch lightning fast_dev_run', default=False)

    return parser.parse_args()


def get_model(args, in_channels, in_length):
    model = UnetVAE(
                in_length = in_length,
                in_channels = in_channels,
                num_residual_layers=args.num_residual_layers,
                num_residual_hiddens=args.num_residual_hiddens,
                num_embeddings=args.num_embeddings,
                embedding_dim=args.embedding_dim,
            )
    return model

def get_classifier(args):
    classifier = ClassifierUnet(
        embedding_dim=args.embedding_dim, 
        num_embeddings=args.num_embeddings,
        n_classes=len(args.class_used)
    )
    return classifier

def get_dataset(args):
    num_workers = min(4, os.cpu_count() // 2)
    dataset = VibrationDataset(
        data_root=args.dataset_root,
        dataset_used=args.dataset_used,
        class_used=args.class_used,
        ch_used=args.channels_used
    )
    caching_path = os.path.join(os.getcwd(),'cached')
    os.makedirs(caching_path, exist_ok=True)
    cached_dataset = CachedDataset(dataset, cache_path=os.path.join(caching_path,'cached.pt'))
    
    # 데이터셋 split 비율(예시: 70:15:15)
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        cached_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)  # 재현성
    )
    
    if args.class_loss == 'focal':
        class_counts = count_classes(dataset)
        class_names = sorted(class_counts.keys())
        counts = np.array([class_counts[c] for c in class_names], dtype=np.float32)
        alpha = 1.0 / counts
        focal_alpha = torch.tensor(alpha, dtype=torch.float32)
    else:
        focal_alpha = None
    sample = dataset[0]
    in_channels, in_length = sample[0].shape
    print(f"Data Shape : {in_channels} x {in_length}")
    return [train_set, val_set, test_set], in_channels, in_length, focal_alpha

def get_accelerator_and_strategy():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # 멀티-GPU: DDP 전략
            accelerator = "gpu"
            strategy = DDPStrategy(find_unused_parameters=True)
            devices = num_gpus
        else:
            # 싱글 GPU
            accelerator = "gpu"
            strategy = None
            devices = 1
    else:
        # CPU
        accelerator = "cpu"
        strategy = None
        devices = 1
    print(f"accelerator : {accelerator}, Startegy : {strategy}, No.Devices : {devices}")
    return accelerator, strategy, devices

def save_model_and_classifier(model, classifier, args, save_root='model_saved'):
    """
    모델과 분류기 state_dict를 저장하는 함수.
    저장 경로와 파일명은 args의 주요 하이퍼파라미터를 기반으로 자동 생성.

    Parameters:
        model: 저장할 PyTorch 모델 객체
        classifier: 저장할 분류기 객체
        args: argparse.Namespace, 하이퍼파라미터 정보 포함
        save_root: str, 저장 디렉터리 (기본값 'model_saved')
    """
    saved_model_name = (
        f'ed_{args.embedding_dim}_ne_{args.num_embeddings}_nrl_{args.num_residual_layers}_nrh_{args.num_residual_hiddens}'
    )
    model_filename = f'{saved_model_name}_model.pth'
    classifier_filename = f'{saved_model_name}_classifier.pth'
    save_path = os.path.join(save_root)
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, model_filename))
    torch.save(classifier.state_dict(), os.path.join(save_path, classifier_filename))
    print(f"모델 저장 완료: {os.path.join(save_path, model_filename)}")
    print(f"분류기 저장 완료: {os.path.join(save_path, classifier_filename)}")


if __name__ == '__main__':
    args = get_args()
    args.use_cache = args.use_cache == 1
    args.only_encoder = args.only_encoder == 1
    args.only_reconstruction = args.only_reconstruction == 1
    args.apply_ttt = args.apply_ttt == 1
    
    # ---------------------------
    # Dataset 준비
    # ---------------------------
    dataset, in_channels, in_length, focal_alpha = get_dataset(args)
    num_workers = min(4, os.cpu_count() // 2)

    train_dataloader = DataLoader(dataset[0], batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset[1], batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(dataset[2], batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    print("Ready for DataLoader")
    # ---------------------------
    # 모델 및 분류기 준비
    # ---------------------------
    model = get_model(args, in_channels, in_length)
    classifier = get_classifier(args)
    print("Ready for models")
    # ---------------------------
    # Trainer 세팅
    # ---------------------------
    trainer_module = Trainer(
        model=model,
        batch_size=args.batch_size,
        recon_loss=args.recon_loss,
        class_loss=args.class_loss,
        focal_alpha=focal_alpha,
        classifier=classifier,
        classes=args.class_used
        )
    print("Ready for Training")
    # ---------------------------
    # Wandb Logger 설정
    # ---------------------------
    if args.run_name is None:
        name = f'UnetVae_embed{args.embedding_dim}_num_emb{args.num_embeddings}'
    else:
        name = args.run_name
    wandb_logger = WandbLogger(
        project=args.project_name,  # 프로젝트 이름
        name=name,
        save_dir='results',
        log_model=True,  # 모델 구조 로깅
        config = vars(args)
    )
    print("Ready for Logging")
    # ---------------------------
    # Lightning Trainer 실행
    # ---------------------------
    profiler = SimpleProfiler()
    accelerator, strategy, devices = get_accelerator_and_strategy()

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=wandb_logger,
        log_every_n_steps=50,
        fast_dev_run=args.fast_dev_run,
        profiler=profiler,
        precision=32
    )
    
    trainer.fit(trainer_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.validate(trainer_module, dataloaders=valid_dataloader)
    trainer.test(trainer_module, dataloaders=test_dataloader)
    
    save_model_and_classifier(model, classifier, args)

    # 실행 후 데이터 로딩 시간이 오래 걸리는지 확인
    print(trainer.profiler.summary())
