import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

class ResNetEnClassify(nn.Module):
    """
    4개의 단일 채널 이미지를 입력으로 받아 분류를 수행하는 ResNet 모델.
    Args:
        num_classes (int): 분류할 클래스의 총 개수.
        image_size (int): 입력 이미지의 높이와 너비. 기본값: 224.
    """
    def __init__(self, res_select: int, num_classes: int, image_size: int = 224, pretrained: bool = False, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = 224 if pretrained else image_size

        # 1. 사전 학습되지 않은 ResNet-18 모델 구조를 로드합니다.
        # weights=None 으로 설정하여 가중치를 무작위로 초기화합니다.
        if res_select == 18:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet = resnet18(weights=weights)
        elif res_select == 50:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.resnet = resnet50(weights=weights)

        # 2) 입력 채널 수정 (기본 resnet은 conv1: (3->64, 7x7, s=2))
        old_conv = self.resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                # 기존 3채널 가중치
                w = old_conv.weight  # (64, 3, 7, 7)
                # 앞 3채널은 그대로 복사
                new_conv.weight[:, :3] = w
                # 추가 채널은 3채널 평균으로 초기화
                mean = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
                new_conv.weight[:, 3:] = mean.repeat(1, 4-3, 1, 1)
        self.resnet.conv1 = new_conv

        # 3) 최종 FC 헤드 교체
        in_feats = self.resnet.fc.in_features  # resnet18/34=512, resnet50/101=2048
        if dropout > 0:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_feats, num_classes)
            )
        else:
            self.resnet.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

# --- 코드 사용 예시 ---
if __name__ == '__main__':
    # 모델 파라미터 설정
    RES_SELECT = 18 # 18 or 50
    NUM_CLASSES = 5
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    PRETRAINED = False # T or F

    # 모델 인스턴스 생성
    model = ResNetEnClassify(res_select=RES_SELECT, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, pretrained=PRETRAINED)

    # 모델 구조 출력
    print("--- Custom ResNet Model Architecture ---")
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