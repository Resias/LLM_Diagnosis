import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', logits=True):
        """
        alpha: 클래스별 가중치 (float or list), 불균형 클래스일 경우 보정
        gamma: focusing 파라미터
        reduction: 'mean', 'sum', 'none'
        logits: 모델 output이 raw logit인지 (True) softmax or sigmoid 결과인지 (False)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            if inputs.shape[1] > 1:
                BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
                probs = torch.softmax(inputs, dim=1)
            else:
                BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
                probs = torch.sigmoid(inputs)
        else:
            BCE_loss = F.nll_loss(torch.log(inputs), targets, reduction='none')
            probs = inputs

        # 정답 클래스에 대한 예측 확률 p_t
        if inputs.shape[1] > 1:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            pt = probs
            targets = targets.float()

        # focal weight
        focal_weight = (1 - pt) ** self.gamma

        # alpha 적용
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        elif isinstance(self.alpha, list):
            alpha_t = torch.tensor(self.alpha).to(inputs.device)[targets]
        else:
            alpha_t = 1

        loss = alpha_t * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ReconLoss(nn.Module):
    def __init__(self, loss_name='mae'):
        super().__init__()
        if loss_name == 'mae':
            self.loss = nn.L1Loss()
        elif loss_name == 'mse':
            self.loss = nn.MSELoss()
        elif loss_name == 'huber':
            self.loss = nn.HuberLoss()
        else:
            assert f'Wrong Loss for Reconstruction : {loss_name}'
    def forward(self, y_hat, y):
        return self.loss(y_hat, y)

class ClassLoss(nn.Module):
    def __init__(self, loss_name='ce', num_classes=5, alpha=None):
        super().__init__()
        self.num_classes = num_classes
        if loss_name=='ce':
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss_name == 'focal':
            self.loss = FocalLoss(alpha=alpha)
        else:
            assert f'Wrong Loss for Classification : {loss_name}'
    def forward(self, y_hat, y):
        return self.loss(y_hat, y)