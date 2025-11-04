import torch
from typing import Callable
from collections import OrderedDict
from functools import partial
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

from data.dataset import OrderInvariantSignalImager, VibrationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class Decoder(nn.Module):
    """Transformer Model Decoder."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        
        # Encoder와 동일한 구조의 Decoder 레이어를 쌓습니다.
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"decoder_layer_{i}"] = EncoderBlock( # EncoderBlock을 그대로 사용
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        # Encoder의 역순으로 연산을 수행합니다.
        # 1. Layer Normalization
        x = self.ln(input)
        # 2. Decoder Layers
        x = self.layers(x)
        # 3. Dropout
        x = self.dropout(x)
        # 4. Positional Embedding 제거 (더하기의 역연산)
        # x = x - self.pos_embedding
        
        return x
    
class VisionTransformerAE(nn.Module):
    def __init__(self,
                num_layers = 12,
                num_heads = 12,
                hidden_dim = 768,
                mlp_dim = 3072,
                dropout = 0.0,
                attention_dropout  = 0.0,
                norm_layer = partial(nn.LayerNorm, eps=1e-6),
                image_size = 224,
                image_channel = 4,
                patch_size = 16,
                masking_ratio=0.75,
                num_classes=5):
        super().__init__()

        seq_length = (image_size // patch_size) ** 2

        self.encoder = Encoder(
            int(seq_length+1),
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.decoder = Decoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.masking_ratio = masking_ratio
        
        self.conv_proj = nn.Conv2d(
                in_channels=image_channel, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        self.conv_transpose_proj = nn.ConvTranspose2d(
            in_channels=hidden_dim, out_channels=image_channel, kernel_size=patch_size, stride=patch_size
        )
        
        self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, self.hidden_dim).normal_(std=0.02))
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        self.heads = nn.Sequential(heads_layers)
        
    def process_patch(self, x: torch.Tensor) -> torch.Tensor:
        n, seq_len, hidden_dim = x.shape
        p = self.patch_size
        n_h = self.image_size // p
        n_w = self.image_size // p
        
        torch._assert(seq_len == n_h * n_w, "Input sequence length must match the number of patches!")
        
        # 1. (n, (n_h * n_w), hidden_dim) -> (n, hidden_dim, (n_h * n_w))
        # _process_input의 permute 역연산
        x = x.permute(0, 2, 1)

        # 2. (n, hidden_dim, (n_h * n_w)) -> (n, hidden_dim, n_h, n_w)
        # _process_input의 reshape 역연산
        x = x.reshape(n, self.hidden_dim, n_h, n_w)
        
        # 3. (n, hidden_dim, n_h, n_w) -> (n, c, h, w)
        # _process_input의 conv_proj 역연산
        x = self.conv_transpose_proj(x)
        
        return x

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def reconstruct(self, img):
        """Masking 없이 이미지 Recon은 잘하는지"""
        # 1. Image -> Patches
        patches = self.process_input(img)
        batch_size = patches.shape[0]
        
        # 2. ★★★ 위치 임베딩을 마스킹 전에 먼저 적용 ★★★
        patches_with_pos = patches + self.pos_embedding
        
        # 3. Encode: class token 추가 후 unmasked patches만 인코딩
        # class token에는 위치 정보가 필요 없습니다 (항상 첫 번째 위치).
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([class_token, patches_with_pos], dim=1)
        encoded_embedding = self.encoder(encoder_input)
        
        # 5. Decode (Decoder 입력 준비)
        # 5.1. 인코더 출력에서 class token 제거
        encoded_patches = encoded_embedding[:, 1:]
        
        # 6. Decode
        decoded_patches = self.decoder(encoded_patches)
        
        # 7. Reconstruct Image
        reconstructed_img = self.process_patch(decoded_patches)
        
        return reconstructed_img
    
    def forward_classify(self, current_img: torch.Tensor, normal_img: torch.Tensor):
        # 1. Image -> Patches
        current_patches = self.process_input(current_img)
        normal_patches = self.process_input(normal_img)
        batch_size = current_patches.shape[0]
        
        # 2. ★★★ 위치 임베딩 ★★★
        current_patches_with_pos = current_patches + self.pos_embedding
        normal_patches_with_pos = normal_patches + self.pos_embedding
        
        # 3. Encode: class token 추가 후 unmasked patches만 인코딩
        # class token에는 위치 정보가 필요 없습니다 (항상 첫 번째 위치).
        current_class_token = self.class_token.expand(batch_size, -1, -1)
        current_encoder_input = torch.cat([current_class_token, current_patches_with_pos], dim=1)
        current_encoded_embedding = self.encoder(current_encoder_input)
        current_class_embedding = current_encoded_embedding[:, 0]
        
        normal_class_token = self.class_token.expand(batch_size, -1, -1)
        normal_encoder_input = torch.cat([normal_class_token, normal_patches_with_pos], dim=1)
        normal_encoded_embedding = self.encoder(normal_encoder_input)
        normal_class_embedding = normal_encoded_embedding[:, 0]
        
        class_embedding = normal_class_embedding - current_class_embedding
        pred = self.heads(class_embedding)
        
        return pred, class_embedding

    def random_masking(self, patch_seq):
        batch_size, seq_len, dim = patch_seq.shape
        num_masked = int(self.masking_ratio * self.seq_length)
        
        # Random Pick Mask Index and Unmask Index
        rand_indices = torch.rand(batch_size, seq_len, device=patch_seq.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        batch_range = torch.arange(batch_size, device=patch_seq.device)[:, None]
        unmasked_patches = patch_seq[batch_range, unmasked_indices]
        
        return unmasked_patches, masked_indices, unmasked_indices
        
    def forward_mae(self, img: torch.Tensor):
        """MAE의 전체 순전파 로직"""
        # 1. Image -> Patches
        patches = self.process_input(img)
        batch_size = patches.shape[0]
        
        # 2. ★★★ 위치 임베딩을 마스킹 전에 먼저 적용 ★★★
        patches_with_pos = patches + self.pos_embedding
        
        # 3. Random Masking (위치 정보가 포함된 패치 대상)
        unmasked_patches, masked_indices, unmasked_indices = self.random_masking(patches_with_pos)
        
        # 4. Encode: class token 추가 후 unmasked patches만 인코딩
        # class token에는 위치 정보가 필요 없습니다 (항상 첫 번째 위치).
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([class_token, unmasked_patches], dim=1)
        encoded_embedding = self.encoder(encoder_input)
        
        # 5. Decode (Decoder 입력 준비)
        # 5.1. 인코더 출력에서 class token 제거
        encoded_patches = encoded_embedding[:, 1:]
        
        # 5.2. 디코더 입력 시퀀스 생성: mask_token에 해당 위치의 pos_embedding을 더해줌
        mask_tokens_with_pos = self.mask_token + self.pos_embedding.repeat(batch_size, 1, 1)[torch.arange(batch_size)[:, None], masked_indices]
        
        # 5.3. 전체 시퀀스 길이로 복원: 인코딩된 패치와 마스크 토큰 결합
        decoder_input = torch.cat([encoded_patches, mask_tokens_with_pos], dim=1)
        
        # 5.4. 원래 순서대로 정렬 (unshuffling)
        combined_indices = torch.cat([unmasked_indices, masked_indices], dim=1)
        unshuffle_indices = combined_indices.argsort(dim=-1)
        decoder_input = decoder_input[torch.arange(batch_size)[:, None], unshuffle_indices]
        
        # 6. Decode
        decoded_patches = self.decoder(decoder_input)
        
        # 7. Reconstruct Image
        reconstructed_img = self.process_patch(decoded_patches)
        
        return reconstructed_img, patches, masked_indices
        
    def calculate_mae_loss(self, reconstructed_img, original_img, masked_indices):
        """오직 마스킹된 패치에 대해서만 MSE Loss를 계산"""
        # 원본 이미지를 패치로 변환
        original_patches = self.process_input(original_img)
        
        # 복원된 이미지를 패치로 변환
        reconstructed_patches = self.process_input(reconstructed_img)

        # 마스킹되었던 패치들만 선택 (Ground Truth)
        batch_range = torch.arange(original_patches.shape[0], device=original_patches.device)[:, None]
        target_patches = original_patches[batch_range, masked_indices]

        # 예측값에서 마스킹되었던 위치의 패치들만 선택 (Prediction)
        predicted_patches = reconstructed_patches[batch_range, masked_indices]
        
        # MSE 손실 계산
        loss = nn.MSELoss()(predicted_patches, target_patches)
        return loss

    def encode(self, x: torch.Tensor):
        # 1. Image -> Patches
        patches = self.process_input(x)
        batch_size = patches.shape[0]
        
        # 2. ★★★ 위치 임베딩 ★★★
        patches_with_pos = patches + self.pos_embedding
        
        # 3. Encode: class token 추가 후 unmasked patches만 인코딩
        # class token에는 위치 정보가 필요 없습니다 (항상 첫 번째 위치).
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([class_token, patches_with_pos], dim=1)
        encoded_embedding = self.encoder(encoder_input)
        class_embedding = encoded_embedding[:, 0]
        
        return class_embedding

if __name__ == '__main__':
    data_root = '/Volumes/dataset_onlyMac/processed'
    data_mode = 'stft+cross'

    signal_imger = OrderInvariantSignalImager(
            mode=data_mode,
            log1p=True,
            normalize="per_channel",  # None | "per_channel" | "global"
            eps=1e-8,
            out_dtype=torch.float32,
            max_order=20.0,           # order 축 상한
            H_out=224,                # order-bin 수
            W_out=224,                # time-bin 수
            # STFT
            stft_nperseg=1024,
            stft_hop=256,
            stft_window="hann",
            stft_center=True,
            stft_power=1.0,           # 1: magnitude, 2: power
        )
    dataset_names = ['dxai', 'mfd','vat','vbl']
    total_dataset = VibrationDataset(
        data_root=data_root,
        using_dataset=dataset_names,
        window_sec=5,
        stride_sec=2,
        cache_mode='file',
        transform=signal_imger
    )
    
    # vit_b_16과 동일한 세팅의 ViT_AE
    # hidden_dim=768
    # vit_l_16
    # 
    vit_ae = VisionTransformerAE(
                    num_layers = 12,
                    num_heads = 12,
                    hidden_dim = 768,
                    mlp_dim = 3072,
                    dropout = 0.0,
                    attention_dropout  = 0.0,
                    norm_layer = partial(nn.LayerNorm, eps=1e-6),
                    image_size = 224,
                    image_channel = 4,
                    patch_size = 16,
                    masking_ratio=0.75,
                    num_classes=5
    )

    num_epoch = 200
    batch_size = 32

    data_loader = DataLoader(
        dataset=total_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(vit_ae.parameters(), lr=1.5e-4, weight_decay=0.05)
    # Classification Loss 설정
    classify_criterion = nn.CrossEntropyLoss()

    for epoch in num_epoch:
        
        epoch_loss_mae = 0.0
        epoch_loss_classify = 0.0
        total_epoch_loss = 0.0
        
        for idx, batch in enumerate(data_loader):
            current_img, current_cls, _, normal_img, _, _ = batch
            
            # MAE forward & Loss 연산
            reconstructed_img, _, masked_indices = vit_ae.forward_mae(img=current_img)
            loss_mae = vit_ae.calculate_mae_loss(reconstructed_img, current_img, masked_indices)
            
            # Classification forward & Loss 연산
            predictions = vit_ae.forward_classify(current_img=current_img, normal_img=normal_img)
            loss_classify = classify_criterion(predictions, current_cls)
            
            
            total_loss = loss_mae + loss_classify
            total_loss.backward()
            optimizer.step()
            
            epoch_loss_mae += loss_mae.item()
            epoch_loss_classify += loss_classify.item()
            total_epoch_loss += total_loss.item()
        
        