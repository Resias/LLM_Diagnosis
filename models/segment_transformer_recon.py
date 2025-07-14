import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segment_transformer import SegmentEmbedder, SegmentSelfAttention, SegmentCrossAttention, SegmentClassifier


class ResidualSegmentSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        # 기존의 SegmentSelfAttention 사용
        self.self_attn = SegmentSelfAttention(embed_dim, n_heads)
        # Residual 이후 정규화를 위한 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, S, D) — B: batch size, S: num segments, D: embed_dim
        Returns:
            out: (B, S, D) — residual+norm 처리된 출력
            attn_weights: (B, S, S) — attention weight 행렬
        """
        # 1) self-attention
        attn_out, attn_weights = self.self_attn(x)  # → (B, S, D), (B, S, S)

        # 2) residual 연결
        res = x + attn_out

        # 3) layer normalization
        out = self.norm(res)
        return out, attn_weights

class DecoderBlock1D(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, 
                 kernel_size:int, stride:int, padding:int=0):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        # x: (B, in_ch, L)
        out = self.convtr(x)
        return out  # (B, out_ch, L')   


class SegmentReconModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 3,
        n_dec_layers: int = 2,
        num_segments: int = 10,
        seg_len: int = 26,
        num_classes: int = 4
    ):
        super().__init__()
        self.num_segments = num_segments
        self.seg_len = seg_len
        self.embed_dim = embed_dim
        # 1) 공유 인코더
        self.embedder = SegmentEmbedder(embed_dim)
        self.enc_layers = nn.ModuleList([
            ResidualSegmentSelfAttention(embed_dim, n_heads)
            for _ in range(n_enc_layers)
        ])

        # 2) 재구성 디코더 (reconstruction head)
        channels = [embed_dim] + [embed_dim // 2] * (n_dec_layers - 2) + [2]
        self.dec_blocks = nn.ModuleList()
        prev_ch = embed_dim
        for i, out_ch in enumerate(channels):
            # 첫 블록만 세그먼트 길이만큼 업샘플
            if i == 0:
                ks, st = seg_len, seg_len
            else:
                ks, st = 1, 1
            self.dec_blocks.append(
                DecoderBlock1D(prev_ch, out_ch, kernel_size=ks, stride=st)
            )
            prev_ch = out_ch

        self.cross_attn = SegmentCrossAttention(embed_dim, n_heads)
        # 3) 분류 헤드 (classification head)
        #    세그먼트 임베딩을 평균(pool)한 뒤 FC
        self.classifier = SegmentClassifier(embed_dim, num_segments, num_classes)

    def forward(self, x_sample, x_normal, classify=False, get_z=False):
        """
        Args:
            x: 원본 FFT 시퀀스, shape (B, 2, seg_len * num_segments)
        Returns:
            recon: 복원된 FFT, (B, 2, seg_len * num_segments)
            logits: 분류 로짓, (B, num_classes)
        """
        
        B = x_sample.size(0)

        # 1) 세그먼트 분할 → (B, S, 2, seg_len)
        segs = x_sample.unfold(2, self.seg_len, self.seg_len).permute(0, 2, 1, 3)
        norm_seg = x_normal.unfold(2, self.seg_len, self.seg_len).permute(0, 2, 1, 3)
        # 2) 임베딩 → (B, S, D)
        sample_embed = self.embedder(segs)
        normal_embed = self.embedder(norm_seg)

        # 3) Self-Attention 인코딩
        sample_attn = sample_embed
        normal_attn = normal_embed
        sample_scores_list = []
        normal_scores_list = []
        for attn in self.enc_layers:
            sample_attn, scores = attn(sample_attn)
            normal_attn, scores = attn(normal_attn)
            sample_scores_list.append(scores)
            normal_scores_list.append(scores)
        if get_z:
            return sample_attn, normal_attn, {
                "sample_attn_scores_list": sample_scores_list,
                "normal_attn_scores_list": normal_scores_list
            }


        # --- 재구성 헤드 ---
        # (B, S, D) → (B, D, S)
        x = sample_attn.permute(0,2,1)
        # 각 디코더 블록 적용
        for dec in self.dec_blocks:
            x = dec(x)
        recon = x

        if classify:
            # --- 분류 헤드 ---
            cross_attn_out, cross_scores = self.cross_attn(sample_attn, normal_attn)  # (B, 10, D), (B, 10, 10)
            logits = self.classifier(cross_attn_out)   # (B, num_classes)
            return recon, logits, {
                "sample_attn_scores_list": sample_scores_list,
                "normal_attn_scores_list": normal_scores_list,
                "cross_attn_scores": cross_scores
            }
        return recon, {
                "sample_attn_scores_list": sample_scores_list,
                "normal_attn_scores_list": normal_scores_list,
                "cross_attn_scores": None
            }


# ---------------------------
# 학습 루프 예시
# ---------------------------
if __name__ == "__main__":
    # 더미 입력: B=4, C=2, L=260
    dummy_x = torch.randn(4, 2, 26 * 10)
    dummy_x2 = torch.randn(4, 2, 26 * 10)
    # 더미 레이블: 분류용
    dummy_y = torch.randint(0, 4, (4,))

    model = SegmentReconModel(
        embed_dim=64,
        n_heads=4,
        n_enc_layers=3,
        n_dec_layers=3,
        num_segments=10,
        seg_len=26,
        num_classes=4
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    # reconstruction 만
    recon, _ = model(dummy_x, dummy_x2)
    print(f"Recon shape: {recon.shape}")  # (B, 2, seg_len * num_segments)
    # classification 시에
    recon, logits, scores = model(dummy_x, dummy_x2, classify=True)
    print(f"Recon shape: {recon.shape}, Logits shape: {logits.shape}")  # (B, 2, seg_len * num_segments), (B, num_classes)
    # latent z를 가져오기위해
    recon_attn, norm_attn, scores = model(dummy_x, dummy_x2, classify=False, get_z=True)
    print(f"Recon Attn shape: {recon_attn.shape}, Norm Attn shape: {norm_attn.shape}")  # (B, S, D)
    loss_recon = recon_criterion(recon, dummy_x)
    loss_cls = cls_criterion(logits, dummy_y)
    # 멀티태스크 가중치: 재구성 0.5, 분류 0.5 (필요시 조정)
    loss = 0.5 * loss_recon + 0.5 * loss_cls

    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"Total Loss: {loss.item():.6f}, Recon: {loss_recon.item():.6f}, "
          f"Cls: {loss_cls.item():.6f}")
