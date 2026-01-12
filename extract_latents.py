import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import torch.nn as nn

from data.dataset import OrderInvariantSignalImager, VibrationDataset
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE   # ← 모델 정의 파일 경로에 맞게 수정

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. 모델 로드
# -----------------------------
def load_model(ckpt_path: str):
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
        num_classes=5
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


# -----------------------------
# 2. Latent 벡터 추출
# -----------------------------
@torch.no_grad()
def extract_latents(model, dataloader, max_batches=None, select=0, device=None):
    latents = []
    labels = []
    for i, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
        x = batch["x_stft"].to(device)
        y = batch["x_cls"].to(device)
        ref_x = batch['ref_stft'].to(device)

        if select == 0:
            z = model.encode(x)   # (B, 768)
        elif select == 1:
            _, z = model.forward_classify(x, ref_x)

        latents.append(z.cpu().numpy())
        labels.append(y.cpu().numpy())

        if max_batches and i >= max_batches:
            break

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    return latents, labels
