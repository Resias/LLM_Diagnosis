import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

CLASS_COLOR = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
}

def plot_pca(latents, labels, save_path=None):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(latents)

    plt.figure(figsize=(7, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=10,
            color=CLASS_COLOR.get(int(cls), "gray"),
            label=f"class {cls}",
            alpha=0.8,
        )
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("PCA of ViT-AE Latent Space")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_tsne(latents, labels, save_path=None):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=42,
    )
    Z = tsne.fit_transform(latents)

    plt.figure(figsize=(7, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=10,
            color=CLASS_COLOR.get(int(cls), "gray"),
            label=f"class {cls}",
            alpha=0.8,
        )
    
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.title("t-SNE of ViT-AE Latent Space")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
