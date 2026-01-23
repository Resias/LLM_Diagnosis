import torch
from extract_latents import load_model, extract_latents
from visualize_latents import plot_pca, plot_tsne
from tokenizer_trainer.visualize import create_reconstruction_figure
from torch.utils.data import DataLoader

from data.dataset import OrderInvariantSignalImager, VibrationDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Dataset
# -----------------------------
signal_imger = OrderInvariantSignalImager(
    mode="stft",
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

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------
# Model & Latents
# -----------------------------

path =[
    '/root/tmp/LLM_Diagnosis/checkpoints/last_model_0_All_dataset_non_masking.pth'
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/last_model_4_ds4_20251209_030620.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/only_recon_model_Except_DXAI.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/last_model_3_ds3_20251203_035437.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/only_recon_model_3_ds3_20251203_035437.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/last_model_2_ds2_20251201_020043.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/only_recon_model_2_ds2_20251201_020043.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/last_model_1_ds1_20251128_023043.pth',
    # '/root/tmp/LLM_Diagnosis/checkpoints_saving_before_260106/only_recon_model_1_ds1_20251128_023043.pth'
]

select = [0] # 0: non-diff latent, 1: diff latent
batch = next(iter(loader))
x = batch['x_stft'].to(device, non_blocking=True)


for p in path:
    for flag in select:
        model = load_model(p, img_ch=2).to(device)

        # latents, labels = extract_latents(model, loader, max_batches=1000, select=flag, device=device)

        # # -----------------------------
        # # Visualization
        # # -----------------------------
        # model_setting = p.split('/')[-1].split('ds')[0]
        # print(f"Processing {model_setting} with flag {flag}...")
        # flag_select = f"non_diff" if flag == 0 else f"diff"
        # plot_pca(latents, labels, save_path=f"{model_setting}_pca_latent_{flag_select}.png")
        # plot_tsne(latents, labels, save_path=f"{model_setting}_tsne_latent_{flag_select}.png")
        
        rec = model.reconstruct(img=x)
        fig = create_reconstruction_figure(orig_tensor=x[0], rec_tensor=rec[0], mode='stft',
                                           max_order='20.0', window_sec='5.0')
        fig.savefig("reconstruction.png", dpi=300, bbox_inches="tight")
