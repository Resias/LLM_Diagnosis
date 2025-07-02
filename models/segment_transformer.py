import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentEmbedder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 26, embed_dim)
        )

    def forward(self, x):
        # x: (batch, 10, 2, 26)
        B, S, C, L = x.shape
        x = x.reshape(B * S, 1, C, L)  # 명시적 reshape
        out = self.embed(x)           # → (B*S, embed_dim)
        out = out.reshape(B, S, -1)   # → (B, 10, embed_dim)
        return out


class SegmentSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, x):
        # x: (batch, 10, embed_dim)
        out, attn_weights = self.attn(x, x, x, need_weights=True)
        return out, attn_weights  # (B, 10, D), (B, 10, 10)


class SegmentCrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, query, keyval):
        # query: (B, 10, D), keyval: (B, 10, D)
        out, attn_weights = self.attn(query, keyval, keyval, need_weights=True)
        return out, attn_weights  # (B, 10, D), (B, 10, 10)


class SegmentClassifier(nn.Module):
    def __init__(self, embed_dim, num_segments, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * num_segments, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(-1)
        )
    def forward(self, x):
        return self.fc(x)

class SegmentLevelModel(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, num_segments=10, num_classes=4, seg_len=26):
        super().__init__()
        self.seg_len = seg_len
        self.embedder = SegmentEmbedder(embed_dim)
        self.self_attn = SegmentSelfAttention(embed_dim, n_heads)
        self.cross_attn = SegmentCrossAttention(embed_dim, n_heads)
        self.classifier = SegmentClassifier(embed_dim, num_segments, num_classes)

    def forward(self, x_sample, x_normal):

        seg_sample = x_sample.unfold(2, self.seg_len, self.seg_len).permute(0, 2, 1, 3)
        seg_normal = x_normal.unfold(2, self.seg_len, self.seg_len).permute(0, 2, 1, 3)

        # x_sample, x_normal: (batch, 10, 2, 26)
        sample_embed = self.embedder(seg_sample)             # (B, 10, D)
        normal_embed = self.embedder(seg_normal)             # (B, 10, D)

        sample_attn, sample_scores = self.self_attn(sample_embed)   # (B, 10, D), (B, 10, 10)
        normal_attn, normal_scores = self.self_attn(normal_embed)   # (B, 10, D), (B, 10, 10)

        cross_attn_out, cross_scores = self.cross_attn(sample_attn, normal_attn)  # (B, 10, D), (B, 10, 10)

        out = self.classifier(cross_attn_out)  # (B, num_classes)

        return out, {
            "sample_attn_scores": sample_scores,
            "normal_attn_scores": normal_scores,
            "cross_attn_scores": cross_scores
        }
