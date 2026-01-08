import torch
from extract_latents import load_model, extract_latents
from visualize_latents import plot_pca, plot_tsne
from torch.utils.data import DataLoader

from data.dataset import OrderInvariantSignalImager, VibrationDataset

flag = 1  # 0: non diff latent, 1: diff latent

# -----------------------------
# Dataset
# -----------------------------
signal_imger = OrderInvariantSignalImager(
    mode="stft+cross",
    log1p=True,
    normalize="per_channel",
    eps=1e-8,
    out_dtype=torch.float32,
    max_order=20.0,
    H_out=224,
    W_out=224,
    stft_nperseg=1024,
    stft_hop=256,
    stft_window="hann",
    stft_center=True,
    stft_power=1.0,
)

dataset = VibrationDataset(
    data_root="data/processed",
    using_dataset=["vat", "vbl", "mfd", "dxai"],
    window_sec=5.0,
    stride_sec=3.0,
    transform=signal_imger,
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# Model & Latents
# -----------------------------
model = load_model("/root/tmp/LLM_Diagnosis/checkpoints/last_model_1_ds1_20260106_112809.pth")

latents, labels = extract_latents(model, loader, max_batches=1000, select=flag)

# -----------------------------
# Visualization
# -----------------------------
plot_pca(latents, labels, save_path=f"pca_latent_{flag}.png")
plot_tsne(latents, labels, save_path=f"tsne_latent_{flag}.png")