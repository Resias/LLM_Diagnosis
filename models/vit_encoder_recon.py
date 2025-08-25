import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights


# ---------------------------
# 유틸: patchify / unpatchify
# ---------------------------
def patchify(img, patch_size):
    # img: (N, C, H, W)
    N, C, H, W = img.shape
    p = patch_size
    assert H % p == 0 and W % p == 0
    h = H // p
    w = W // p
    # (N, C, h, p, w, p) -> (N, h*w, p*p*C)
    x = img.reshape(N, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(N, h * w, p * p * C)
    return x  # (N, L, patch_dim)

def unpatchify(tokens, channels, img_size, patch_size):
    # tokens: (N, L, p*p*C)
    N, L, D = tokens.shape
    p = patch_size
    C = channels
    H = W = img_size
    h = H // p
    w = W // p
    assert L == h * w and D == p * p * C
    x = tokens.reshape(N, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.reshape(N, C, H, W)
    return x

# ---------------------------
# MAE-style 복원 디코더 블록
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x

class MAEDecoder(nn.Module):
    """
    MAE 스타일의 얕은 디코더.
    - encoder_dim -> decoder_dim로 사상
    - 마스크 토큰 주입 + 디코더 포지셔널 임베딩
    - 몇 개의 Transformer 블록 후 각 패치 픽셀 회복
    """
    def __init__(self, encoder_dim, decoder_dim, num_layers, num_heads,
                 num_patches, patch_dim, image_size, patch_size):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))  # +1 for cls
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_dim)
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, z_vis, ids_restore):
        """
        z_vis: (N, L_vis, encoder_dim)  - encoder 출력(클래스 토큰 제외, 보이는 패치들만)
        ids_restore: (N, L)             - 원래 순서로 되돌리기 위한 인덱스
        """
        N, L = ids_restore.shape
        z = self.decoder_embed(z_vis)  # (N, L_vis, D)
        # 마스크 토큰을 만들어 붙인다 (L_mask = L - L_vis)
        L_vis = z.shape[1]
        L_mask = L - L_vis
        mask_tokens = self.mask_token.expand(N, L_mask, -1)
        # [보이는 토큰 | 마스크 토큰]을 원래 패치 순서로 복원
        z_ = torch.cat([z, mask_tokens], dim=1)  # (N, L, D) in shuffled order
        z_ = torch.gather(z_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1]))  # (N, L, D)

        # 디코더용 cls 토큰은 0으로(선택), pos_embed는 +1(cls)
        cls_tok = torch.zeros(N, 1, z_.shape[-1], device=z_.device, dtype=z_.dtype)
        z_ = torch.cat([cls_tok, z_], dim=1)  # (N, L+1, D)
        z_ = z_ + self.pos_embed[:, : z_.shape[1], :]

        for blk in self.blocks:
            z_ = blk(z_)
        z_ = self.norm(z_)
        z_ = z_[:, 1:, :]  # cls 제거

        # 패치별 픽셀 예측 (N, L, patch_dim)
        pred = self.pred(z_)
        return pred  # (N, L, p*p*C)

class VITEnClassify(nn.Module):
    """
    4개의 단일 채널 이미지를 입력으로 받아 분류를 수행하는 Vision Transformer 모델.

    Args:
        num_classes (int): 분류할 클래스의 총 개수.
        image_size (int): 입력 이미지의 높이와 너비. 기본값: 224.
        patch_size (int): 이미지를 나눌 패치의 크기. 기본값: 16.
    """
    def __init__(self, num_classes: int, image_size: int = 224, model_size: str = 'b',
                 pretrained: bool = False, mask_ratio: float = 0.75,
                 dec_dim: int = 512, dec_layers: int = 4, dec_heads: int = 8, patch_size: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = 224 if pretrained else image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 1. ViT 모델 구조를 로드합니다.
        if model_size == 'b':
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.vit = vit_b_16(weights=weights, image_size=self.image_size)
            enc_dim = 768
        elif model_size == 'l':
            weights = ViT_L_16_Weights.DEFAULT if pretrained else None
            self.vit = vit_l_16(weights=weights, image_size=self.image_size)
            enc_dim = 1024
        else:
            raise ValueError("model_size arguments must be 'b' or 'l'")

        # 2. ViT의 첫 번째 레이어인 패치 프로젝션(Conv2d)을 4채널 입력에 맞게 수정합니다.
        # 기존 Conv2d의 출력 채널 수(hidden_dim)와 다른 파라미터는 유지합니다.
        original_conv_proj = self.vit.conv_proj
        assert isinstance(original_conv_proj, nn.Conv2d), "conv_stem_configs를 쓰지 않는 기본 ViT만 가정"

        hidden_dim = original_conv_proj.out_channels
        k = original_conv_proj.kernel_size[0]
        s = original_conv_proj.stride[0]
        new_conv = nn.Conv2d(4, hidden_dim, kernel_size=k, stride=s, bias=(original_conv_proj.bias is not None))
        with torch.no_grad():
            if original_conv_proj.weight.shape[1] == 3:
                # 3->4채널 확장: 마지막 채널을 평균으로 초기화
                new_conv.weight[:, :3] = original_conv_proj.weight
                new_conv.weight[:, 3:] = original_conv_proj.weight.mean(dim=1, keepdim=True)
                if original_conv_proj.bias is not None:
                    new_conv.bias.copy_(original_conv_proj.bias)
            else:
                new_conv.weight.copy_(original_conv_proj.weight)
                if original_conv_proj.bias is not None:
                    new_conv.bias.copy_(original_conv_proj.bias)
        self.vit.conv_proj = new_conv

        # 3. ViT의 최종 분류기(Head)를 우리의 클래스 개수에 맞게 수정합니다.
        self.vit.heads = nn.Linear(in_features=enc_dim, out_features=self.num_classes)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        nn.init.trunc_normal_(self.enc_mask_token, std=0.02)
        
        # 4) 복원 디코더 구성(MAE 스타일)
        num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = self.patch_size * self.patch_size * 4
        self.decoder = MAEDecoder(
            encoder_dim=enc_dim, decoder_dim=dec_dim, num_layers=dec_layers, num_heads=dec_heads,
            num_patches=num_patches, patch_dim=patch_dim, image_size=self.image_size, patch_size=self.patch_size
        )

        # 디버깅용
        self.num_patches = num_patches
        self.enc_dim = enc_dim
        self.patch_dim = patch_dim

    # ---------------------------
    # 내부: 인코더 전방(전체 토큰)
    # ---------------------------
    def _encode_full(self, x):
        # torchvision 내부: _process_input -> cls concat -> encoder -> cls 토큰 추출
        x_seq = self.vit._process_input(x)                 # (N, L, enc_dim)
        cls = self.vit.class_token.expand(x_seq.shape[0], 1, -1)
        x_seq = torch.cat([cls, x_seq], dim=1)            # (N, 1+L, enc_dim)
        z = self.vit.encoder(x_seq)                       # (N, 1+L, enc_dim)
        return z                                           # cls 포함

    # ---------------------------
    # 내부: 인코더 전방(마스크 적용, 보이는 토큰만)
    # ---------------------------
    @torch.no_grad()
    def _random_mask(self, N, L, device):
        # MAE 방식: 패치 인덱스 셔플 후 앞쪽 L_vis만 사용
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=device)  # (N, L)
        ids_shuffle = torch.argsort(noise, dim=1)  # 작은 값이 keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return ids_keep, ids_restore, len_keep

    def _encode_masked(self, x):
        # x -> patch 임베딩까지만 얻어 마스크/가시 토큰 분기
        x_seq = self.vit._process_input(x)  # (N, L, enc_dim)
        N, L, D = x_seq.shape
        device = x.device

        ids_keep, ids_restore, len_keep = self._random_mask(N, L, device)
        # 가시 토큰만 인코더에 통과 (클래스 토큰 추가)
        x_vis = torch.gather(x_seq, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))  # (N, L_vis, D)
        
        # 3) encoder 입력 길이를 L로 맞추기: 마스크 토큰 채워넣고 원래 순서로 unshuffle
        L_mask = L - x_vis.shape[1]
        mask_tokens = self.enc_mask_token.expand(N, L_mask, D)                    # (N, L_mask, D)
        x_full = torch.cat([x_vis, mask_tokens], dim=1)                            # (N, L, D) (셔플 순서)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))  # (N, L, D)

        cls_ = self.vit.class_token.expand(N, 1, -1)
        z_full = self.vit.encoder(torch.cat([cls_, x_full], dim=1))                 # pos_embedding 길이와 일치

        # 5) 가시 토큰 위치의 출력만 꺼내 디코더로
        z_full_wo_cls = z_full[:, 1:, :]                                           # (N, L, D)
        z_vis = torch.gather(z_full_wo_cls, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))  # (N, L_vis, D)
        return z_vis, ids_restore, ids_keep

    # ---------------------------
    # forward: 공동학습용 출력
    # ---------------------------
    def forward(self, x):
        """
        반환:
          logits: (N, num_classes)
          rec_pred: (N, C, H, W)         # 복원 이미지
          aux: dict(ids_restore, rec_tokens, mask_ratio)
        """
        # 1) 분류 경로(전체 토큰)
        z_full = self._encode_full(x)             # (N, 1+L, D)
        logits = self.vit.heads(z_full[:, 0, :])  # cls -> head

        # 2) 복원 경로(마스크 토큰/디코더)
        z_vis, ids_restore, ids_keep = self._encode_masked(x)        # (N, L_vis, D), (N, L)
        rec_tokens = self.decoder(z_vis, ids_restore)       # (N, L, patch_dim)
        rec_img = unpatchify(rec_tokens, 4, self.image_size, self.patch_size)  # (N, C, H, W)

        return logits, rec_img, {"ids_restore": ids_restore, "ids_keep": ids_keep,
                         "mask_ratio": self.mask_ratio, "rec_tokens": rec_tokens}

# --- 코드 사용 예시 ---
if __name__ == '__main__':
    # 모델 파라미터 설정
    NUM_CLASSES = 5
    IMAGE_SIZE = 224
    BATCH_SIZE = 16

    # 모델 인스턴스 생성
    model = VITEnClassify(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)

    # 모델 구조 출력
    print("--- Custom ViT Model Architecture ---")
    print(model)
    print("------------------------------------")


    # 더미(dummy) 입력 데이터 생성
    # 4개의 (B, 4, H, W) 형태의 텐서를 생성
    dummy_img = torch.randn(BATCH_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)

    # 모델의 forward pass 실행
    output_logits = model(dummy_img)

    # ====== 핵심 수정: 언팩 ======
    logits, rec_img, aux = model(dummy_img)

    # 출력 형태 확인
    print(f"\nInput shape: {tuple(dummy_img.shape)}")            # (8, 4, 256, 256)
    print(f"Logits shape: {tuple(logits.shape)}")               # (8, 5)
    print(f"Reconstruction shape: {tuple(rec_img.shape)}")      # (8, 4, 256, 256)

    # 보조 정보
    ids_restore = aux["ids_restore"]     # (N, L)
    ids_keep = aux["ids_keep"]           # (N, L_vis)
    print(f"ids_restore shape: {tuple(ids_restore.shape)}")
    print(f"ids_keep shape: {tuple(ids_keep.shape)}")

    # 간단 검증
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert rec_img.shape == (BATCH_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE)
    print("Model forward pass successful!")