#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference.py — VisionTransformerAE 추론 스크립트
- 학습 코드와 동일한 STFT/이미징 파이프라인 재사용
- 단일 GPU, AMP(bfloat16/float16) 자동 사용
- 분류 확률/예측 CSV 저장 (+옵션: 리컨 이미지/로짓 NPY 저장)
"""

import os
import sys
import math
import argparse
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === 프로젝트 의존 모듈 (학습 코드와 동일 경로 가정) ===
from data.dataset import VibrationDataset, OrderInvariantSignalImager, CachedDataset
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE

# ----------------------------- 유틸 -----------------------------

def _to_uint8_image(t: torch.Tensor):
    """
    t: (C,H,W) float/half -> uint8 HxW(또는 HxW,3)
    """
    t = t.detach().float().cpu()
    if t.dim() == 3:  # C,H,W
        if t.size(0) >= 3:
            img = t[:3]
        else:
            img = t.mean(0, keepdim=True).repeat(3, 1, 1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # H,W,3
        return img
    elif t.dim() == 2:  # H,W
        img = (t - t.min()) / (t.max() - t.min() + 1e-8)
        img = (img * 255.0).clamp(0, 255).byte().numpy()
        return img
    else:
        raise ValueError("Unexpected tensor shape for image.")

def vib_collate(batch):
    """
    학습 코드와 동일: 모델이 바로 쓰는 키만 stack, 나머지는 리스트로 유지
    """
    out = {}
    must_stack = {"x_stft", "ref_stft", "x_cls"}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if k in must_stack:
            vals = [torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in vals]
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out

def load_checkpoint(model: nn.Module, ckpt_path: str, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    meta = {
        "epoch": ckpt.get("epoch", None),
        "val_acc": ckpt.get("val_acc", None),
        "optimizer_state_keys": list(ckpt.get("optimizer_state_dict", {}).keys()) if isinstance(ckpt, dict) else []
    }
    return meta

# ----------------------------- 추론 루프 -----------------------------

@torch.no_grad()
def run_inference(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # AMP 설정
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 출력 경로
    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"pred_{run_tag}.csv")
    logits_path = os.path.join(args.output_dir, f"logits_{run_tag}.npy") if args.save_logits_npy else None
    recon_dir = os.path.join(args.output_dir, f"recon_{run_tag}") if args.save_recon_images else None
    if recon_dir:
        os.makedirs(recon_dir, exist_ok=True)

    # -------- 데이터셋 구성 (학습 코드 로직과 동일한 분기) --------
    data_root_abs = os.path.join(os.getcwd(), args.data_root)
    sample = None
    if args.cached:
        selected = args.dataset_select
        if selected == 0:
            test_root = os.path.join(data_root_abs, "llm_vib_validset_4dataset_only_dxai.pt")
        elif selected == 1:
            test_root = os.path.join(data_root_abs, "llm_vib_validset_only_vat.pt")
        elif selected == 2:
            test_root = os.path.join(data_root_abs, "llm_vib_validset_only_vbl.pt")
        elif selected == 3:
            test_root = os.path.join(data_root_abs, "llm_vib_validset_only_mfd.pt")
        elif selected == 4:
            test_root = os.path.join(data_root_abs, "llm_vib_validset_only_dxai.pt")
        else:
            raise ValueError("Unexpected selected caching dataset index.")
        dataset = CachedDataset(data_root=test_root)
        # CachedDataset이 x_stft/ref_stft/x_cls 등을 동일 키로 제공한다고 가정
        signal_imger = None
    else:
        # 이미지 변환기
        output_size = 224 if args.pretrained else args.image_size
        signal_imger = OrderInvariantSignalImager(
            mode=args.stft_mode,
            log1p=True,
            normalize="per_channel",
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
        )
        # 기본은 dxai 검증셋과 동일 분포를 가정
        using_dataset = args.using_dataset.split(",") if args.using_dataset else ['dxai']
        dataset = VibrationDataset(
            data_root=data_root_abs,
            using_dataset=using_dataset,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            transform=signal_imger
        )
    sample = dataset[0]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, min(4, (os.cpu_count() or 8) // 2)),
        pin_memory=True,
        collate_fn=vib_collate
    )

    # -------- 모델 구성 (학습 코드와 동일 하이퍼파라미터 기본값) --------
    model = VisionTransformerAE(
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.0,
        attention_dropout=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        image_size=224,
        image_channel=4,
        patch_size=16,
        masking_ratio=0.75,
        num_classes=args.num_classes,
    ).to(device)
    model.eval()

    # 체크포인트 로드
    meta = load_checkpoint(model, args.checkpoint, map_location=device)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint} (epoch={meta.get('epoch')}, val_acc={meta.get('val_acc')})")

    # -------- 추론 --------
    all_rows = []
    all_logits = [] if args.save_logits_npy else None

    softmax = nn.Softmax(dim=1)

    # 저장할 샘플 수 제한(리컨 이미지)
    recon_save_count = 0
    recon_save_limit = args.max_recon_images

    for bidx, batch in enumerate(loader):
        x = batch['x_stft'].to(device, non_blocking=True)
        ref_x = batch['ref_stft'].to(device, non_blocking=True) if 'ref_stft' in batch else None
        # 라벨이 있을 수도/없을 수도 있음
        y = batch.get('x_cls', None)
        if isinstance(y, torch.Tensor):
            y = y.to(device, non_blocking=True)

        # 파일/메타 정보가 있다면 함께 저장
        # dataset 구현에 따라 키가 다를 수 있으므로 안전하게 처리
        ids = batch.get('id', None) or batch.get('idx', None) or list(range(bidx * args.batch_size, bidx * args.batch_size + x.size(0)))
        paths = batch.get('path', [''] * x.size(0))

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            # 분류
            logits, diff = model.forward_classify(current_img=x, normal_img=ref_x)
            probs = softmax(logits)
            pred = torch.argmax(probs, dim=1)

            # (옵션) MAE 리컨 생성 및 저장
            if args.save_recon_images and recon_save_count < recon_save_limit:
                rec, _, _ = model.forward_mae(img=x)
                # 배치 일부만 저장
                k = min(rec.size(0), recon_save_limit - recon_save_count)
                for i in range(k):
                    orig_img = _to_uint8_image(x[i])
                    rec_img = _to_uint8_image(rec[i])
                    # H,W,3 가정
                    try:
                        import imageio.v2 as imageio
                    except Exception:
                        import imageio
                    # 원본/리컨 나눠 저장
                    base = f"b{bidx:05d}_i{i:03d}"
                    imageio.imwrite(os.path.join(recon_dir, f"{base}_orig.png"), orig_img)
                    imageio.imwrite(os.path.join(recon_dir, f"{base}_recon.png"), rec_img)
                recon_save_count += k

        # 결과 누적
        probs_np = probs
        pred_np = pred
        logits_np = logits

        if all_logits is not None:
            all_logits.append(logits_np)

        # CSV 행 생성
        for i in range(x.size(0)):
            row = {
                "id": ids[i] if isinstance(ids, list) else ids[i].item() if torch.is_tensor(ids) else ids,
                "path": paths[i] if isinstance(paths, list) else paths,
                "pred": int(pred_np[i]),
            }
            if y is not None:
                row["label"] = int(y[i].item())
            # 각 클래스 확률
            for c in range(probs_np.shape[1]):
                row[f"p_{c}"] = float(probs_np[i, c])
            all_rows.append(row)

    # 저장
    print(sample)
    print( f"[INFO] Inference completed on {len(loader.dataset)} samples.")
    print( f"[INFO] Calculating accuracy..." )
    print( f"[RESULT] Accuracy: ", end='' )
    accuracy = sum(1 for r in all_rows if 'label' in r and r['pred'] == r['label']) / sum(1 for r in all_rows if 'label' in r) if any('label' in r for r in all_rows) else 'N/A'
    print(accuracy)
    print( f"[INFO] Error Result" )
    print(1-accuracy if accuracy != 'N/A' else 'N/A' )
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved predictions: {csv_path} (rows={len(df)})")

    if all_logits is not None:
        logits_all = np.concatenate(all_logits, axis=0) if len(all_logits) > 0 else np.zeros((0, args.num_classes), dtype=np.float32)
        np.save(logits_path, logits_all)
        print(f"[OK] Saved logits: {logits_path} shape={logits_all.shape}")

    if recon_dir:
        print(f"[OK] Saved reconstruction samples in: {recon_dir} (total saved <= {recon_save_count})")

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inference for VisionTransformerAE on vibration images")

    # 데이터/전처리
    p.add_argument('--data_root', type=str, default='data/processed', help='Processed data root')
    p.add_argument('--cached', action='store_true', help='Use CachedDataset .pt (same split rules as training)')
    p.add_argument('--dataset_select', type=int, default=4, help='0: all, 1: except vat, 2: except vbl, 3: except mfd, 4: only dxai(valid)')
    p.add_argument('--using_dataset', type=str, default='dxai,vbl', help='non-cached일 때 사용할 데이터셋들, 콤마 구분 (예: "dxai" 또는 "vat,vbl,mfd,dxai")')

    p.add_argument('--window_sec', type=float, default=5.0)
    p.add_argument('--stride_sec', type=float, default=3.0)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--max_order', type=float, default=20.0)

    p.add_argument('--stft_mode', type=str, default='stft+cross', choices=['stft', 'stft+cross', 'stft_complex'])
    p.add_argument('--stft_nperseg', type=int, default=1024)
    p.add_argument('--stft_hop', type=int, default=256)
    p.add_argument('--stft_power', type=float, default=1.0)

    # 모델/체크포인트
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--pretrained', action='store_true', help='(참고) 학습 시 ImageNet 프리트레인 사용 여부. 추론 입력 크기 결정용 플래그.')
    p.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 체크포인트(.pth). 보통 checkpoints/best_model.pth')

    # 실행/출력
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--output_dir', type=str, default='inference_outputs')
    p.add_argument('--tag', type=str, default='', help='출력 파일명 태그')

    # (옵션) 저장
    p.add_argument('--save_logits_npy', action='store_true', help='로짓을 .npy로 저장')
    p.add_argument('--save_recon_images', action='store_true', help='MAE 리컨 결과 일부 PNG 저장')
    p.add_argument('--max_recon_images', type=int, default=50, help='저장할 리컨 샘플 최대 수')

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
