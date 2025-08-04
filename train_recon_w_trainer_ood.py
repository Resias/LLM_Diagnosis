import os
import argparse
import torch
import lightning as L
import time

from tqdm import tqdm

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from data.order_dataset import OrderFreqDataset, CachedDataset, LightningDM
from models.segment_transformer_recon import SegmentReconModel
from models.trainer_recon_classify_ood import LightningReconClassifyMD
from callbacks.attn_vis import AttnVisCallback

import wandb

TOTAL_DATASET=['dxai', 'mfd', 'vat', 'vbl']

def get_args():
    parser = argparse.ArgumentParser(description="Train Config")
    # Wandb Settings
    parser.add_argument('--project_name', type=str, help='Wandb project Name', default='TEST')
    parser.add_argument('--run_name', type=str, help='Wandb experiments Name', default=None)

    # Datset Settings
    parser.add_argument('--use_cache', type=int, help='use cached dataset', choices=[0, 1], default=1)
    parser.add_argument('--dataset_root', type=str, help='for raw data csv root', default='/workspace/vms_dataset')
    parser.add_argument('--classes', type=list, help='classes for training', default=['normal', 'looseness', 'misalignment', 'unbalance', 'bearing'])
    parser.add_argument('--ood_dataset', type=str, choices=TOTAL_DATASET, default=None)
    parser.add_argument('--dataset_list', type=list, help='datasets for training', default=['dxai', 'mfd', 'vat', 'vbl'])
    parser.add_argument('--average_size', type=int, help='average smoothing size', default=100)
    parser.add_argument('--target_length', type=int, help='data shape for final length', default=260)
    parser.add_argument('--sensor_list', type=list, help='sensor name for trainining', default=['motor_x', 'motor_y'])
    parser.add_argument('--max_order', type=int, help='max order for data preprocessing', default=10)
    parser.add_argument('--cache_dir', type=str, help='for preprocessed cached data saving path', default='./cache')
    # Cached Dataset Setting
    parser.add_argument('--cached_dataset', type=str, help='for all preprocessed cached data saving path', default='cached_dataset.pt')
    # Lightning Data Module Settings
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--data_splits_ratio', type=list, help='train, validation or plus test dataset ratio', default=[0.7, 0.3])
    parser.add_argument('--training_mode', type=str, help='for training mode', default='recon_only') # 'recon_classify' or 'recon_only'
    parser.add_argument('--class_loss', type=str, help='for classification loss', default='focal')
    parser.add_argument('--recon_loss', type=str, help='for reconstruction loss', default='huber')
    parser.add_argument('--cont_loss', type=str, help='for contrastive loss', default='sup')
    parser.add_argument('--loss_alpha', type=float, help='for classification loss alpha', default=0.5)
    parser.add_argument('--cont_threshold', type=float, help='for contrastive starting threshold', default=0.75)
    parser.add_argument('--cont_loss_alpha', type=float, help='for contrastive loss alpha', default=0.1)
    parser.add_argument('--cont_temp', type=float, help='for contrastive temperature', default=0.1)
    # Model Settings
    parser.add_argument('--embed_dim', type=int, help='model embed dimension', default=64)
    parser.add_argument('--n_heads', type=int, help='model num of heads', default=16)
    parser.add_argument('--n_enc_layers', type=int, help='model num of encoder layer', default=6)
    parser.add_argument('--n_dec_layers', type=int, help='model num of decoder layer', default=6)
    parser.add_argument('--n_segments', type=int, help='model num of segments', default=10)
    parser.add_argument('--num_classes', type=int, help='model num of classes', default=4)
    parser.add_argument('--positional_encode', type=bool, help='Setting about postional Encoding', default=False)
    # Training Settings
    parser.add_argument('--fast_dev_run', type=bool, help='pytorch lightning fast_dev_run', default=False)
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs', default=200)
    return parser.parse_args()



def get_dataset(args, stage: str):
    num_workers = min(2, os.cpu_count() // 2)
    
    preprocess_dataset = OrderFreqDataset(
        data_root= args.dataset_root,
        classes = args.classes, 
        dataset_list= args.dataset_list,
        averaging_size = args.average_size, 
        target_len = args.target_length, 
        sensor_list = args.sensor_list,
        max_order = args.max_order
    )
    print("Getting preprocessed Dataset")
    # cached = CachedDataset(dataset=preprocess_dataset)
    dm = LightningDM(
        args,
        dataset = preprocess_dataset,
        batch_size = args.batch_size,
        seed = args.seed,
        splits = args.data_splits_ratio,
        dataset_list= [d for d in TOTAL_DATASET if d not in args.dataset_list],
        classes = args.classes
    )
    print("Ready for Setup")
    dm.setup(stage)
    if args.class_loss == 'focal':
        focal_alpha = dm.focal_alpha
    else:
        focal_alpha = None
    print("DataSet Ready")
    return dm, focal_alpha
    
def get_model(args, focal_alpha=None):
    model = SegmentReconModel(
        embed_dim = args.embed_dim,
        n_heads = args.n_heads,
        n_enc_layers = args.n_enc_layers,
        n_dec_layers = args.n_dec_layers,
        num_segments = args.n_segments,
        num_classes = len(args.classes)
    )
    Lmd = LightningReconClassifyMD(
        model = model,
        training_mode = args.training_mode,
        recon_loss = args.recon_loss,
        class_loss = args.class_loss,
        cont_loss = args.cont_loss,
        loss_alpha = args.loss_alpha,
        cont_threshold = args.cont_threshold,
        cont_alpha = args.cont_loss_alpha,
        cont_temperature = args.cont_temp,
        focal_alpha = focal_alpha,
        classes = args.classes,
        postional_enc= args.positional_encode
    )
    return Lmd

if __name__ == '__main__':
    L.seed_everything(42, workers=True)
    args = get_args()
    args.use_cache = args.use_cache == 1
    if args.ood_dataset is not None:
        args.dataset_list = [d for d in TOTAL_DATASET if d != args.ood_dataset]
    else:
        # fallback: 사용자가 직접 dataset_list 줬다면 그대로
        pass
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()

    datamodule, focal_alpha = get_dataset(args, 'fit')
    model = get_model(args, focal_alpha)

    num_epoch = args.max_epochs
    # WandB Logger 설정
    if args.run_name is None:
        now = time.localtime()
        name = f'MD_{now.tm_mon}{now.tm_mday}_ed{args.embed_dim}_nh{args.n_heads}_nd{args.n_dec_layers}_ne{args.n_enc_layers}'
    else:
        name = args.run_name

    wandb_logger = WandbLogger(
        project=args.project_name,  # 프로젝트 이름
        name = name,
        save_dir = 'results',
        log_model = True,  # 모델 구조 로깅
        config = vars(args)
    )
    if num_gpus >= 2:
        strategy = DDPStrategy(find_unused_parameters=True)
        accelerator = 'gpu'
        devices = num_gpus
    elif num_gpus == 1:
        strategy = None
        accelerator = 'gpu'
        devices = 1
    else:
        strategy = None
        accelerator = 'cpu'
        devices = 1
    
    vis_cb = AttnVisCallback(max_samples_per_stage=4000, use_tsne=True, tsne_perplexity=30)

    trainer = L.Trainer(
        callbacks=[vis_cb],
        max_epochs = args.max_epochs,
        accelerator = accelerator,
        devices = devices,
        strategy = strategy,
        logger = wandb_logger,
        log_every_n_steps = 1,
        fast_dev_run = args.fast_dev_run,
        precision=32
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    
    saved_model_name = f'MD_{now.tm_mon}{now.tm_mday}_ed{args.embed_dim}_nh{args.n_heads}_nd{args.n_dec_layers}_ne{args.n_enc_layers}'
    model_filename = f'{saved_model_name}_model.pth'
    save_path = os.path.join('model_saved')
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, model_filename))
    wandb.finish()