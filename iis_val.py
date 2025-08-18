import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.vit_encoder import VITEnClassify
from data.dataset import WindowedVibrationDataset, OrderInvariantSignalImager
import wandb
import ast
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_loader)
    
    # 전체 정확도 계산
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    return val_loss, accuracy, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description='Validate ViT model on IIS dataset')
    parser.add_argument('--data_root', type=str, default='data/processed',
                        help='Path to the processed data directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--window_sec', type=float, default=5.0)
    parser.add_argument('--stride_sec', type=float, default=2.0)
    parser.add_argument('--max_order', type=float, default=20.0)
    parser.add_argument('--stft_mode', type=str, default='stft+cross',
                        choices=['stft', 'stft+cross', 'stft_complex'])
    parser.add_argument('--stft_nperseg', type=int, default=1024)
    parser.add_argument('--stft_hop', type=int, default=256)
    parser.add_argument('--stft_power', type=float, default=1.0)
    parser.add_argument('--project_name', type=str, default='vibration-diagnosis-iis-val')
    args = parser.parse_args()

    # wandb 초기화
    wandb.init(project=args.project_name, config=vars(args))
    
    # 데이터 준비
    data_root = os.path.join(os.getcwd(), args.data_root)
    meta_csv = os.path.join(data_root, 'meta.csv')
    meta_pd = pd.read_csv(meta_csv)
    
    meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
    meta_pd = meta_pd[5 <= meta_pd['data_sec']]
    
    # IIS 데이터셋만 선택
    meta_pd = meta_pd[meta_pd['dataset'] == 'iis']
    print(f'Selected {len(meta_pd)} samples from IIS dataset')
    
    # 이미지 변환기 설정
    signal_imger = OrderInvariantSignalImager(
        mode=args.stft_mode,
        log1p=True,
        normalize="per_channel",
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=args.max_order,
        H_out=args.image_size,
        W_out=args.image_size,
        stft_nperseg=args.stft_nperseg,
        stft_hop=args.stft_hop,
        stft_window="hann",
        stft_center=True,
        stft_power=args.stft_power,
    )
    
    # 데이터셋 생성
    dataset = WindowedVibrationDataset(
        meta_df=meta_pd,
        data_root=data_root,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        cache_mode='file',
        transform=signal_imger
    )
    
    # 데이터로더 생성
    val_loader = DataLoader(dataset, 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          num_workers=4,
                          pin_memory=True)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 생성 및 가중치 로드
    model = VITEnClassify(
        num_classes=args.num_classes,
        image_size=args.image_size,
        patch_size=16
    ).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} "
              f"with validation accuracy {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    # Loss function 설정
    criterion = nn.CrossEntropyLoss()
    
    # Validation 실행
    val_loss, accuracy, all_preds, all_labels = validate_model(model, val_loader, criterion, device)
    
    # 결과 출력 및 로깅
    print(f"\nValidation Results on IIS Dataset:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['normal', 'unbalance', 'looseness', 'misalignment', 'bearing']
    
    # Classification Report 출력
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # wandb에 결과 로깅
    wandb.log({
        'val_loss': val_loss,
        'val_accuracy': accuracy,
        'confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })
    
    # 클래스별 정확도 로깅
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        class_accuracy = 100. * np.mean(np.array(all_preds)[class_mask] == i)
        wandb.log({f'accuracy_{class_name}': class_accuracy})
    
    wandb.finish()

if __name__ == "__main__":
    main()
