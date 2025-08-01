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

class LightningReconClassifyMD(L.LightningModule):
    def __init__(self, 
                 model,
                 training_mode='recon_classify',
                 recon_loss='mse',
                 class_loss='ce',
                 loss_alpha=0.5,
                 focal_alpha=None,
                 classes=None,
                 postional_enc = False):
        super(LightningReconClassifyMD, self).__init__()
        self.model = model
        self.recon_loss = ReconLoss(loss_name=recon_loss)
        self.class_loss = ClassLoss(loss_name=class_loss, alpha=focal_alpha)
        self.classes = classes
        self.y_train_true, self.y_train_pred = [], []
        self.y_valid_true, self.y_valid_pred = [], []
        self.y_test_true, self.y_test_pred = [], []
        self.first_val_step = True
        self.automatic_optimization = True
        self.training_mode = training_mode  # 'recon_classify' or 'recon_only'
        self.alpha = loss_alpha
        self.position_enc = postional_enc
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
        signal_data, normal_tensor, rms, y = batch
        NanChecking = [torch.isnan(signal_data).any(), torch.isnan(normal_tensor).any(), torch.isnan(y).any()]
        if any(NanChecking):
            raise ValueError(
                f"NaN detected in batch data. Breaking."
                f"Signal has Nan: {NanChecking[0]}"
                f"Normal has Nan: {NanChecking[1]}"
                f"y has Nan: {NanChecking[2]}")
        
        logger_step = self.global_step if hasattr(self, 'global_step') else batch_idx
        if self.training_mode == 'recon_only':
            recon, attn = self.model(signal_data, normal_tensor)
            NanChecking = [torch.isnan(recon).any()]
            if any(NanChecking):
                raise ValueError(
                    f"Recon has Nan: {NanChecking[0]}"
                    f", batch_idx={batch_idx}")
            
            loss = self.recon_loss(recon, signal_data)
            pred = None
            self.log(f"train/recon_loss", loss, sync_dist=True)
            self.log_attention_heatmaps_random(attn, pred, y, "train", logger_step, num_samples=2)
        elif self.training_mode == 'recon_classify':
            recon, pred, attn = self.model(signal_data, normal_tensor, classify=True, positional_encode = self.position_enc)
            NanChecking = [torch.isnan(recon).any(), torch.isnan(pred).any()]
            if any(NanChecking):
                raise ValueError(
                    f"Recon has Nan: {NanChecking[0]}"
                    f", Pred has Nan: {NanChecking[1]}"
                    f", batch_idx={batch_idx}")
            
            rec_loss = self.recon_loss(recon, signal_data)
            cl_loss = self.class_loss(pred, y)

            self.y_train_true.extend(y.cpu().numpy())
            self.y_train_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
            self.log(f"train/recon_loss", rec_loss, sync_dist=True)
            self.log(f"train/class_loss", cl_loss, sync_dist=True)

            self.log_attention_heatmaps_random(attn, pred, y, "train", logger_step, num_samples=2)

            loss = self.alpha * rec_loss + (1 - self.alpha) * cl_loss
            self.log(f"train/total_loss", loss, sync_dist=True)
        
        # 로그: 원본 vs 재구성
        self.log_reconstructions_random(signal_data, recon, "train", logger_step, num_samples=1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        signal_data, normal_tensor, rms, y = batch
        NanChecking = [torch.isnan(signal_data).any(), torch.isnan(normal_tensor).any(), torch.isnan(y).any()]
        if any(NanChecking):
            raise ValueError(
                f"NaN detected in batch data. Breaking."
                f"Signal has Nan: {NanChecking[0]}"
                f"Normal has Nan: {NanChecking[1]}"
                f"y has Nan: {NanChecking[2]}")
        logger_step = self.global_step if hasattr(self, 'global_step') else batch_idx

        if self.training_mode == 'recon_only':
            recon, attn = self.model(signal_data, normal_tensor)
            NanChecking = [torch.isnan(recon).any()]
            if any(NanChecking):
                raise ValueError(
                    f"Recon has Nan: {NanChecking[0]}"
                    f", batch_idx={batch_idx}")
            
            loss = self.recon_loss(recon, signal_data)
            pred = None
            self.log(f"valid/recon_loss", loss, sync_dist=True)
            self.log_attention_heatmaps_random(attn, pred, y, "valid", logger_step, num_samples=1)
        elif self.training_mode == 'recon_classify':
            recon, pred, attn = self.model(signal_data, normal_tensor, classify=True, positional_encode = self.position_enc)
            NanChecking = [torch.isnan(recon).any(), torch.isnan(pred).any()]
            if any(NanChecking):
                raise ValueError(
                    f"Recon has Nan: {NanChecking[0]}"
                    f", Pred has Nan: {NanChecking[1]}"
                    f", batch_idx={batch_idx}")
            
            rec_loss = self.recon_loss(recon, signal_data)
            cl_loss = self.class_loss(pred, y)

            self.y_valid_true.extend(y.cpu().numpy())
            self.y_valid_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
            self.log(f"valid/recon_loss", rec_loss, sync_dist=True)
            self.log(f"valid/class_loss", cl_loss, sync_dist=True)

            self.log_attention_heatmaps_random(attn, pred, y, "valid", logger_step, num_samples=2)

            loss = self.alpha * rec_loss + (1 - self.alpha) * cl_loss
            self.log(f"valid/total_loss", loss, sync_dist=True)
        
        # 로그: 원본 vs 재구성
        self.log_reconstructions_random(signal_data, recon, "valid", logger_step, num_samples=2)
        return loss
    
    def test_step(self, batch, batch_idx):
        signal_data, normal_tensor, rms, y = batch
        NanChecking = [torch.isnan(signal_data).any(), torch.isnan(normal_tensor).any(), torch.isnan(y).any()]
        if any(NanChecking):
            raise ValueError(
                f"NaN detected in batch data. Breaking."
                f"Signal has Nan: {NanChecking[0]}"
                f"Normal has Nan: {NanChecking[1]}"
                f"y has Nan: {NanChecking[2]}")
        
        logger_step = self.global_step if hasattr(self, 'global_step') else batch_idx
        if self.training_mode == 'recon_only':
            recon, attn = self.model(signal_data, normal_tensor)
            NanChecking = [torch.isnan(recon).any()]
            if any(NanChecking):
                raise ValueError(
                    f"Recon has Nan: {NanChecking[0]}"
                    f", batch_idx={batch_idx}")
            
            loss = self.recon_loss(recon, signal_data)
            pred = None
            self.log(f"test/recon_loss", loss, sync_dist=True)
            self.log_attention_heatmaps_random(attn, pred, y, "test", logger_step, num_samples=2)
            return loss
        elif self.training_mode == 'recon_classify':
            recon, pred, attn = self.model(signal_data, normal_tensor, classify=True, positional_encode = self.position_enc)
            NanChecking = [torch.isnan(recon).any(), torch.isnan(pred).any()]
            if any(NanChecking):
                raise ValueError(
                    f"Pred has Nan: {NanChecking[0]}"
                    f", batch_idx={batch_idx}")
            
            rec_loss = self.recon_loss(recon, signal_data)
            cl_loss = self.class_loss(pred, y)

            self.y_test_true.extend(y.cpu().numpy())
            self.y_test_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
            self.log(f"test/recon_loss", rec_loss, sync_dist=True)
            self.log(f"test/class_loss", cl_loss, sync_dist=True)
            self.log_attention_heatmaps_random(attn, pred, y, "test", logger_step, num_samples=2)

            loss = self.alpha * rec_loss + (1 - self.alpha) * cl_loss
            self.log(f"test/total_loss", loss, sync_dist=True)
            return loss

    def on_train_epoch_end(self):
        if self.training_mode == 'recon_classify':
            self.classify_report(phase='train')
        self.y_train_true.clear()
        self.y_train_pred.clear()

    def on_validation_epoch_end(self):
        if self.training_mode == 'recon_classify':
            self.classify_report(phase="valid")
        self.y_valid_true.clear()
        self.y_valid_pred.clear()
        
    def on_test_epoch_end(self):
        if self.training_mode == 'recon_classify':
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
        
        attn_keys = ["sample_attn_scores_list", "normal_attn_scores_list", "cross_attn_scores"]
        if preds is not None:
            B = preds.shape[0]
            pred_classes = torch.argmax(preds, dim=1).detach().cpu().numpy()
            true_labels = labels.detach().cpu().numpy()
        else:
            first = next(iter(attn_dict.values()))
            if isinstance(first, list):
                B = first[0].shape[0]
            else:
                B = first.shape[0]
        idx_list = self.select_random_indices(B, num_samples=num_samples)
        
        for idx in idx_list:
            for key in attn_keys:
                scores = attn_dict[key]
                if scores is None:
                    # 이 키에 대한 어텐션 스코어가 없으면 건너뛰기
                    continue
                # 리스트면 각 레이어별로
                if isinstance(scores, list):
                    for layer_i, layer_scores in enumerate(scores):
                        # layer_scores: (B, S, S)
                        attn_map = layer_scores[idx].detach().cpu().numpy()
                        self._plot_and_log(
                            attn_map,
                            f"{step_name}/{key}/layer{layer_i+1}",
                            preds, labels, idx, step_name
                        )
                else:
                    # cross_attn_scores: (B, S, S)
                    attn_map = scores[idx].detach().cpu().numpy()
                    self._plot_and_log(
                        attn_map,
                        f"{step_name}/{key}",
                        preds, labels, idx, step_name
                    )

    def _plot_and_log(self, attn_map, name, preds, labels, idx, step_name):
        import matplotlib.pyplot as plt
        import numpy as np
        import wandb
        plt.figure(figsize=(4,4))
        plt.imshow(attn_map, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xlabel("Key/Value")
        plt.ylabel("Query")
        title = name
        caption = name + f" | epoch:{self.current_epoch}"
        if preds is not None:
            pred_cls = torch.argmax(preds, dim=1)[idx].item()
            true_cls = labels[idx].item()
            title += f" pred{pred_cls}/true{true_cls}"
            caption += f" | pred:{pred_cls} true:{true_cls}"
        plt.title(title)
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log({
                name: wandb.Image(plt.gcf(), caption=caption)
            })
        plt.close()
        gc.collect()
        
    def log_reconstructions_random(self, originals, reconstructions, step_name, logger_step, num_samples=1):
        """
        originals: torch.Tensor (B, C, L)  원본 배치
        reconstructions: torch.Tensor (B, C, L)  재구성 배치
        """
        B, C, L = originals.shape
        idx_list = random.sample(range(B), num_samples)
        for idx in idx_list:
            fig, axes = plt.subplots(C, 1, figsize=(6, 2*C), sharex=True)
            for ch in range(C):
                ax = axes[ch] if C > 1 else axes
                ax.plot(originals[idx, ch].detach().cpu().numpy(), label="original")
                ax.plot(reconstructions[idx, ch].detach().cpu().numpy(), label="recon", alpha=0.7)
                ax.set_ylabel(f"ch{ch}")
                if ch == 0:
                    ax.legend(loc="upper right")
            plt.suptitle(f"{step_name} recon vs orig | epoch={self.current_epoch}")
            plt.xlabel("time / order bin")
            
            # W&B 로깅
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log({
                    f"{step_name}/reconstruction": wandb.Image(fig)
                })
            plt.close(fig)
            gc.collect()