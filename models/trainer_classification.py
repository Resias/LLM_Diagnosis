import wandb
import random

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
        self.automatic_optimization = True
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
        self.log(f"train/loss", loss, sync_dist=True)

        logger_step = self.global_step if hasattr(self, 'global_step') else batch_idx
        self.log_attention_heatmaps_random(attn, pred, y, "train", logger_step, num_samples=2)

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
        self.log(f"valid/loss", loss, sync_dist=True)

        logger_step = self.global_step if hasattr(self, 'global_step') else batch_idx
        self.log_attention_heatmaps_random(attn, pred, y, "valid", logger_step, num_samples=2)

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
        self.log(f"test/loss", loss, sync_dist=True)
        
        logger_step = self.global_step if hasattr(self, 'global_step') else batch_idx
        self.log_attention_heatmaps_random(attn, pred, y, "test", logger_step, num_samples=2)

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
        
    def select_random_indices(self, batch_size, num_samples=2):
        return random.sample(range(batch_size), num_samples)
        
    def log_attention_heatmaps_random(self, attn_dict, preds, labels, step_name, logger_step, num_samples=2):
        """
        attn_dict: {"sample_attn_scores": (B, 10, 10), ...}
        preds: (B, num_classes) softmax output
        labels: (B,) long tensor
        step_name: "train", "valid", "test"
        logger_step: 현재 step
        """
        max_epochs = self.trainer.max_epochs if hasattr(self.trainer, "max_epochs") else 50
        allowed_epochs = [0, max_epochs//4, max_epochs // 2, max_epochs//2 + max_epochs//4, max_epochs - 1]
        if self.current_epoch not in allowed_epochs:
            return
        
        attn_keys = ["sample_attn_scores", "normal_attn_scores", "cross_attn_scores"]
        B = preds.shape[0]
        idx_list = self.select_random_indices(B, num_samples=num_samples)

        pred_classes = torch.argmax(preds, dim=1).detach().cpu().numpy()
        true_labels = labels.detach().cpu().numpy()

        for idx in idx_list:
            for key in attn_keys:
                attn_map = attn_dict[key][idx].detach().cpu().numpy()  # (10, 10)

                plt.figure(figsize=(5, 5))
                plt.imshow(attn_map, cmap="viridis")
                plt.colorbar()

                plt.xticks(ticks=np.arange(0, 10), labels=np.arange(1, 11))
                plt.yticks(ticks=np.arange(0, 10), labels=np.arange(1, 11))

                
                if key == "cross_attn_scores":
                    plt.xlabel("Key/Value (1~10)")
                    plt.ylabel("Query (1~10)")
                
                plt.title(
                    f"{key}_pred{pred_classes[idx]}_true{true_labels[idx]}"
                )
                
                # 🟩 wandb.Image에 caption으로 결과 연동하여 가독성 향상
                caption = f"{step_name} | {key} | pred: {pred_classes[idx]} | true: {true_labels[idx]} | epoch: {self.current_epoch}"


                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.log({
                        f"{step_name}/{key}_heatmap": wandb.Image(plt.gcf(), caption=caption)
                    })
                plt.close()