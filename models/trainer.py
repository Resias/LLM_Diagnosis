import wandb
import gc

import lightning as L

import torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils.loss import ClassLoss, ReconLoss

class LightningMD(L.LightningModule):
    def __init__(self, 
                 model,
                 class_loss='ce',
                 focal_alpha=None,
                 classes=None):
        super(LightningMD, self).__init__()
        self.model = model
        self.class_loss = ClassLoss(loss_name=class_loss, alpha=focal_alpha)
        self.classes = classes
        self.y_train_true, self.y_train_pred = [], []
        self.y_valid_true, self.y_valid_pred = [], []
        self.y_test_true, self.y_test_pred = [], []
        self.first_val_step = True
        self.automatic_optimization = False
        self.save_hyperparameters()

        
    def label_to_index(self, labels):
        """Convert labels to indices."""
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        return torch.tensor([class_to_idx[label] for label in labels], device=self.device)
    
    def configure_optimizers(self):
        opt_model = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        sch_model = torch.optim.lr_scheduler.StepLR(opt_model, step_size=10, gamma=0.5)
        
        return (
            {
                "optimizer": opt_model,
                "lr_scheduler": {
                    "scheduler": sch_model
                }
            }
        )
    
    def training_step(self, batch, batch_idx):
        signal_data, normal_tensor, y = batch
        NanChecking = [torch.isnan(signal_data).any(), torch.isnan(normal_tensor).any(), torch.isnan(y).any()]
        if any(NanChecking):
            raise ValueError(
                f"NaN detected in batch data. Breaking."
                f"Signal has Nan: {NanChecking[0]}"
                f"Normal has Nan: {NanChecking[1]}"
                f"y has Nan: {NanChecking[2]}")
        
        pred, attn = self.model(signal_data, normal_tensor)
        NanChecking = [torch.isnan(pred).any()]
        if any(NanChecking):
            raise ValueError(
                f"Pred has Nan: {NanChecking[0]}"
                f", batch_idx={batch_idx}")
        
        loss = self.class_loss(pred, y)

        self.y_train_true.extend(y.cpu().numpy())
        self.y_train_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
        
        if not self.automatic_optimization:
            opt_model = self.optimizers()
            lr_scheduler_model = self.lr_schedulers()
            
            # 1. Zero grad
            opt_model.zero_grad()

            # 2. Backward
            self.manual_backward(loss)

            # 3. Step
            opt_model.step()
            
            # Scheduler (매 step마다 step 하거나 조건 걸기)
            lr_scheduler_model.step()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        signal_data, normal_tensor, y = batch
        NanChecking = [torch.isnan(signal_data).any(), torch.isnan(normal_tensor).any(), torch.isnan(y).any()]
        if any(NanChecking):
            raise ValueError(
                f"NaN detected in batch data. Breaking."
                f"Signal has Nan: {NanChecking[0]}"
                f"Normal has Nan: {NanChecking[1]}"
                f"y has Nan: {NanChecking[2]}")
        
        pred, attn = self.model(signal_data, normal_tensor)
        NanChecking = [torch.isnan(pred).any()]
        if any(NanChecking):
            raise ValueError(
                f"Pred has Nan: {NanChecking[0]}"
                f", batch_idx={batch_idx}")
        
        loss = self.class_loss(pred, y)

        self.y_valid_true.extend(y.cpu().numpy())
        self.y_valid_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
        return loss
    
    def test_step(self, batch, batch_idx):
        signal_data, normal_tensor, y = batch
        NanChecking = [torch.isnan(signal_data).any(), torch.isnan(normal_tensor).any(), torch.isnan(y).any()]
        if any(NanChecking):
            raise ValueError(
                f"NaN detected in batch data. Breaking."
                f"Signal has Nan: {NanChecking[0]}"
                f"Normal has Nan: {NanChecking[1]}"
                f"y has Nan: {NanChecking[2]}")
        
        pred, attn = self.model(signal_data, normal_tensor)
        NanChecking = [torch.isnan(pred).any()]
        if any(NanChecking):
            raise ValueError(
                f"Pred has Nan: {NanChecking[0]}"
                f", batch_idx={batch_idx}")
        
        loss = self.class_loss(pred, y)

        self.y_test_true.extend(y.cpu().numpy())
        self.y_test_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        self.classify_report(phase='train')
        self.y_train_true.clear()
        self.y_train_pred.clear()

    def on_validation_epoch_end(self):
        self.classify_report(phase="valid")
        self.y_valid_true.clear()
        self.y_valid_pred.clear()
        
    def on_test_epoch_end(self):
        self.classify_report(phase='test')
        self.y_test_true.clear()
        self.y_test_pred.clear()

    def classify_report(self, phase='train'):
        if phase == "train":
            y_true, y_pred = self.y_train_true, self.y_train_pred
        elif phase == "valid":
            y_true, y_pred = self.y_valid_true, self.y_valid_pred
        elif phase == "test":
            y_true, y_pred = self.y_test_true, self.y_test_pred
        
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f'[{phase}] Warning: No predictions, skip metrics.')
            return
        if phase in ['train', 'test']:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
        else:
            unique_classes = np.unique(y_true)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=unique_classes, average='macro', zero_division=1)
        self.log(f"{phase}/Precision", precision, sync_dist=True)
        self.log(f"{phase}/Recall", recall, sync_dist=True)
        self.log(f"{phase}/F1_score", f1, sync_dist=True)
        if getattr(self, "global_rank", 0) == 0:
            self.log_confusion_matrix(y_true, y_pred, epoch_type=phase)
    
    def log_confusion_matrix(self, y_true, y_pred, epoch_type='val'):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.classes)))
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log({f"{epoch_type}/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        gc.collect()