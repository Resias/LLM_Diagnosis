import wandb

import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

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


class Trainer(L.LightningModule):
    def __init__(self, model, batch_size,
                    recon_loss='mae',
                    class_loss='ce',
                    focal_alpha=None,
                    classifier=None,
                    classes=None,
                    ttt_step=3,
                    ttt_lr=1e-5,
                    anomaly_threshold=0.5,
                    apply_ttt=False,
                    only_encoder=False,
                    only_reconstruction=False):
        super(Trainer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.mse_loss = ReconLoss(loss_name=recon_loss)
        self.class_loss = ClassLoss(loss_name=class_loss, alpha=focal_alpha)
    
        self.classifier = classifier
        self.classes = classes
        self.y_true, self.y_pred = [], []
        self.first_val_step = True  # 첫 번째 validation step을 체크하는 플래그
        self.only_encoder = only_encoder
        self.only_reconstruction = only_reconstruction
        
        self.ttt_steps = ttt_step
        self.ttt_lr = ttt_lr
        self.anomaly_threshold = anomaly_threshold
        self.apply_ttt = apply_ttt
        if self.only_encoder:
            print('This model only classify!')
        self.automatic_optimization = False
        
    def label_to_index(self, labels):
        """Convert labels to indices."""
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        return torch.tensor([class_to_idx[label] for label in labels], device=self.device)
    
    def configure_optimizers(self):
        opt_model = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        opt_classifier = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

        sch_model = torch.optim.lr_scheduler.StepLR(opt_model, step_size=10, gamma=0.5)
        sch_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_classifier,
            T_0=50,
            T_mult=25,
            eta_min=1e-5
        )

        return [opt_model, opt_classifier], [sch_model, sch_classifier]

    def compute_recon_loss(self, signal_data, meta_data, bypass_detach=False):
        recon, distribution_loss = self.model(signal_data)
        origianl_distribution_loss = distribution_loss
        recon_loss = self.mse_loss(recon, signal_data)
        
        beta = max(0.0, min(0.1, 1 - abs((self.current_epoch - 100) / 100)))
        total_loss = recon_loss + beta * distribution_loss
        distribution_loss = beta * distribution_loss

        total_loss = total_loss.sum()
        
        return total_loss, recon_loss, origianl_distribution_loss
    
    def compute_classification_loss(self, signal_data, meta_data, ref_data):
        z_input = self.model.encode(signal_data, True)
        z_ref = self.model.encode(ref_data, True)
        pred = self.classifier(z_input, z_ref)
        
        indices = self.label_to_index(meta_data['class_name'])
        
        return self.class_loss(pred, indices), pred, indices
    
    def training_step(self, batch, batch_idx):
        signal_data, meta_data, paried_data, paried_meta_data = batch
        if not self.only_encoder:
            if not self.only_reconstruction:
                total_loss, recon_loss, distribution_loss = self.compute_recon_loss(signal_data, meta_data)
            else:
                _, recon_loss, _ = self.compute_recon_loss(signal_data, meta_data)
                total_loss = recon_loss
        else:
            total_loss = 0.0

        class_loss, pred, indices = self.compute_classification_loss(signal_data, meta_data, ref_data=paried_data)
        
        total_loss += class_loss
        self.y_true.extend(indices.cpu().numpy())
        self.y_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
        
        if not self.automatic_optimization:
            opt_model, opt_classifier = self.optimizers()
            lr_scheduler_model, lr_scheduler_cls = self.lr_schedulers()
            
            # 1. Zero grad
            opt_model.zero_grad()
            opt_classifier.zero_grad()

            # 2. Backward
            self.manual_backward(total_loss)

            # 3. Step
            opt_model.step()
            opt_classifier.step()
            
            # Scheduler (매 step마다 step 하거나 조건 걸기)
            lr_scheduler_model.step()
            lr_scheduler_cls.step()
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        signal_data, meta_data, paried_data, paried_meta_data = batch
        if not self.only_encoder:
            if not self.only_reconstruction:
                total_loss, recon_loss, distribution_loss = self.compute_recon_loss(signal_data, meta_data)
            else:
                _, recon_loss, _ = self.compute_recon_loss(signal_data, meta_data)
                total_loss = recon_loss
        else:
            total_loss = 0.0

        class_loss, pred, indices = self.compute_classification_loss(signal_data, meta_data, ref_data=paried_data)        
        total_loss += class_loss
        self.y_true.extend(indices.cpu().numpy())
        self.y_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """
        test 배치에서 test time training (TTT)을 수행하는 함수.
        학습 시 보지 않은 e 클래스에 대해, 재구성 및 분류 손실을 최소화하도록 인코더와 분류기 일부를 미세 업데이트하고,
        업데이트 후 anomaly score를 계산하여 최종 예측에 반영한다.
        
        ttt_steps: TTT 업데이트 횟수
        ttt_lr: TTT용 학습률
        """
        signal_data, meta_data, paried_data, paried_meta_data = batch
        if not self.only_encoder:
            if not self.only_reconstruction:
                total_loss, recon_loss, distribution_loss = self.compute_recon_loss(signal_data, meta_data)
            else:
                _, recon_loss, _ = self.compute_recon_loss(signal_data, meta_data)
                total_loss = recon_loss
        else:
            total_loss = 0.0

        class_loss, pred, indices = self.compute_classification_loss(signal_data, meta_data, ref_data=paried_data)
        
        total_loss += class_loss
        ty_pred = torch.argmax(pred, dim=1).cpu().numpy()
        anomaly_score = 1 - ty_pred

        # 만약 anomaly score가 임계치보다 높으면 해당 샘플을 looseness 클래스 인덱스는 0 으로 강제 예측
        if self.apply_ttt:
            final_pred = ty_pred.copy()
            for i in range(len(anomaly_score)):
                if anomaly_score[i] > self.anomaly_threshold:
                    final_pred[i] = 0 # looseness 클래스 인덱스는 0
        else:
            final_pred = ty_pred
        self.y_true.extend(indices.cpu().numpy())
        self.y_pred.extend(final_pred)
        
        return total_loss

    def on_train_epoch_end(self):
        self.classify_report(phase='train')

    def on_validation_epoch_end(self):
        if self.trainer.state.fn == "fit":
            prefix = "known_val"
        # 직접 호출한 validate
        elif self.trainer.state.fn == "validate":
            prefix = "unknown_val"
        else:
            prefix = "val"
        self.classify_report(phase=prefix)
        
    def on_test_epoch_end(self):
        self.classify_report(phase='test')

    def classify_report(self, phase='train'):
        if phase in ['train', 'test']:
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro', zero_division=1)
        else:
            unique_classes = np.unique(self.y_true)
            precision, recall, f1, _ = precision_recall_fscore_support( self.y_true, self.y_pred, labels=unique_classes, average='macro', zero_division=1)
        self.log(f"{phase}/precision", precision, self.batch_size, sync_dist=True)
        self.log(f"{phase}/recall", recall, self.batch_size, sync_dist=True)
        self.log(f"{phase}/f1_score", f1, self.batch_size, sync_dist=True)
        if getattr(self, "global_rank", 0) == 0:
            self.log_confusion_matrix(epoch_type=phase)
        self.y_true.clear()
        self.y_pred.clear()
    
    def log_confusion_matrix(self, epoch_type='val'):
        cm = confusion_matrix(self.y_true, self.y_pred, labels=range(len(self.classes)))
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        self.logger.experiment.log({f"{epoch_type}/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)