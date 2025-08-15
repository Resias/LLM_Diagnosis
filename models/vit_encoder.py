import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class VITEnClassify(nn.Module):
    """
    4개의 단일 채널 이미지를 입력으로 받아 분류를 수행하는 Vision Transformer 모델.

    Args:
        num_classes (int): 분류할 클래스의 총 개수.
        image_size (int): 입력 이미지의 높이와 너비. 기본값: 256.
        patch_size (int): 이미지를 나눌 패치의 크기. 기본값: 16.
    """
    def __init__(self, num_classes: int, image_size: int = 256, patch_size: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size

        # 1. 사전 학습되지 않은 ViT-Base 모델 구조를 로드합니다.
        # weights=None 으로 설정하여 가중치를 무작위로 초기화합니다.
        self.vit = vit_b_16(weights=None, image_size=self.image_size)

        # 2. ViT의 첫 번째 레이어인 패치 프로젝션(Conv2d)을 4채널 입력에 맞게 수정합니다.
        # 기존 Conv2d의 출력 채널 수(hidden_dim)와 다른 파라미터는 유지합니다.
        original_conv_proj = self.vit.conv_proj
        hidden_dim = original_conv_proj.out_channels

        self.vit.conv_proj = nn.Conv2d(
            in_channels=4,
            out_channels=hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # 3. ViT의 최종 분류기(Head)를 우리의 클래스 개수에 맞게 수정합니다.
        # ViT-Base 모델의 hidden_dim은 768입니다.
        self.vit.heads = nn.Linear(in_features=hidden_dim, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        4개의 (B, 1, H, W) 텐서를 입력받아 로짓(logits)을 반환합니다.

        Args:
            x (torch.Tensor): (배치 크기, 4, 이미지 높이, 이미지 너비) 형태의 텐서.

        Returns:
            torch.Tensor: (배치 크기, num_classes) 형태의 로짓 텐서.
        """
        # 수정된 ViT 모델에 결합된 텐서를 통과시킵니다.
        logits = self.vit(x)
        return logits

# --- 코드 사용 예시 ---
if __name__ == '__main__':
    # 모델 파라미터 설정
    NUM_CLASSES = 5
    IMAGE_SIZE = 256
    BATCH_SIZE = 8

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

    # 출력 형태 확인
    print(f"\nInput shape (each): ({BATCH_SIZE}, 4, {IMAGE_SIZE}, {IMAGE_SIZE})")
    print(f"Output logits shape: {output_logits.shape}") # 예상 출력: (BATCH_SIZE, NUM_CLASSES)
    assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
    print("Model forward pass successful!")