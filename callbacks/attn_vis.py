# callbacks/attn_vis.py
import torch
import numpy as np
import pandas as pd
import plotly.express as px

from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import lightning as L

import wandb

class AttnVisCallback(L.Callback):
    def __init__(self,
                 max_samples_per_stage=4000,
                 every_n_steps=50,          # 얼마나 자주 로깅/축소할지
                 use_tsne=True,
                 tsne_perplexity=40,
                 viz_modes=("2d","3d")):
        self.max_samples_per_stage = max_samples_per_stage
        self.every_n_steps = every_n_steps
        self.use_tsne = use_tsne
        self.tsne_perplexity = tsne_perplexity
        # stage별 버퍼: {"train": [(z_cross,z_norm,y), ...], "val": [...], "test": [...]}
        self.viz_modes = set(viz_modes)
        self.buf = defaultdict(list)
        self.color_map = None   # 라벨별 고정 색상용

    # ----------------------
    # 내부 유틸
    
    # -------- collect --------
    def _collect(self, stage, pl_module, batch):
        # 배치 구조 맞게 수정: (signal, normal, rms, y)
        if len(batch) == 4:
            signal, normal, rms, y = batch
        else:  # 혹시 (x,y) 형태면 fallback
            signal, y = batch
            normal = signal

        device = pl_module.device
        signal = signal.to(device)
        normal = normal.to(device)

        with torch.no_grad():
            # sample_attn, normal_attn, cross_z, normal_cross_z, _
            _, _, cross_z, _, _ = pl_module.model(
                signal, normal, get_z=True
            )
            # 분류 결과
            if getattr(pl_module, "training_mode", "recon_only") == "recon_classify":
                _, logits, _ = pl_module.model(signal, normal, classify=True)
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                y_pred = np.full(len(y), -1)  # 분류 안 하는 경우 표시용

        # (B,S,D) -> (B,D) : 세그먼트 평균 (원하면 flatten/cls 선택)
        cross_z = cross_z.reshape(cross_z.size(0), -1).cpu().numpy()
        y_true = y.cpu().numpy()

        cur = list(zip(cross_z, y_true, y_pred))
        self.buf[stage].extend(cur)
        if len(self.buf[stage]) > self.max_samples_per_stage:
            self.buf[stage] = self.buf[stage][-self.max_samples_per_stage:]
    
    def _to_str(self, idx, classes):
        if isinstance(idx, (np.integer, int)):
            return classes[idx] if classes else f"cls_{idx}"
        return str(idx)

    def _project_and_log(self, stage, trainer, global_step):
        if trainer.global_rank != 0 or not self.buf[stage]:
            return

        data = self.buf[stage]
        cross_arr_Z = np.stack([d[0] for d in data], axis=0)  # (N,D)
        y_truei = np.array([d[1] for d in data], dtype=int)
        y_predi = np.array([d[2] for d in data], dtype=int)
        correct = (y_truei == y_predi).astype(int)

        classes = getattr(trainer.lightning_module, "classes", None)
        y_true_s = [self._to_str(i, classes) for i in y_truei]
        y_pred_s = [self._to_str(i, classes) if i >= 0 else "none" for i in y_predi]
        
        
        # 색상맵 한 번만 생성
        if self.color_map is None:
            uniq = sorted(list(set(y_true_s)))
            palette = px.colors.qualitative.Safe + px.colors.qualitative.Vivid + px.colors.qualitative.Set3
            self.color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(uniq)}

        log_dict = {}
        if "2d" in self.viz_modes:
            # PCA
            pca = PCA(n_components=2, random_state=42)
            Z_pca = pca.fit_transform(cross_arr_Z)
            df_pca = pd.DataFrame({
                "x": Z_pca[:, 0],
                "y": Z_pca[:, 1],
                "label_true": y_true_s,
                "label_pred": y_pred_s,
                "correct": correct
            })
            fig_pca2d = px.scatter(
                df_pca, x="x", y="y",
                color="label_true",
                symbol="correct",  # True/False로 모양 구분
                hover_data=["label_true", "label_pred"],
                color_discrete_map=self.color_map,
                title=f"{stage} PCA (epoch {trainer.current_epoch})"
            )


            tsne = TSNE(
                n_components=2,
                perplexity=min(self.tsne_perplexity, len(cross_arr_Z) - 1),
                init="random",
                random_state=42
            )
            Z_tsne2 = tsne.fit_transform(cross_arr_Z)
            df_tsne = pd.DataFrame({
                "x": Z_tsne2[:, 0],
                "y": Z_tsne2[:, 1],
                "label_true": y_true_s,
                "label_pred": y_pred_s,
                "correct": correct
            })
            fig_tsne2d = px.scatter(
                df_tsne, x="x", y="y",
                color="label_true",
                symbol="correct",
                hover_data=["label_true", "label_pred"],
                color_discrete_map=self.color_map,
                title=f"{stage} t-SNE (epoch {trainer.current_epoch})"
            )

            log_dict[f"{stage}/pca_2d"] = fig_pca2d
            log_dict[f"{stage}/tsne_2d"] = fig_tsne2d

        # 3D 시각화
        if "3d" in self.viz_modes:
            # PCA 3D
            pca3 = PCA(n_components=3, random_state=42)
            Z_pca3 = pca3.fit_transform(cross_arr_Z)
            df_pca3 = pd.DataFrame({
                "x": Z_pca3[:, 0],
                "y": Z_pca3[:, 1],
                "z": Z_pca3[:, 2],
                "label_true": y_true_s,
                "label_pred": y_pred_s,
                "correct": correct
            })
            fig_pca3 = px.scatter_3d(
                df_pca3, x="x", y="y", z="z",
                color="label_true",
                symbol="correct",
                hover_data=["label_true", "label_pred"],
                color_discrete_map=self.color_map,
                title=f"{stage} PCA 3D (epoch {trainer.current_epoch})"
            )
            log_dict[f"{stage}/pca_3d"] = fig_pca3

            # t-SNE 3D
            tsne3 = TSNE(
                n_components=3,
                perplexity=min(self.tsne_perplexity, len(cross_arr_Z) - 1),
                init="random",
                random_state=42
            )
            Z_tsne3 = tsne3.fit_transform(cross_arr_Z)
            df_tsne3 = pd.DataFrame({
                "x": Z_tsne3[:, 0],
                "y": Z_tsne3[:, 1],
                "z": Z_tsne3[:, 2],
                "label_true": y_true_s,
                "label_pred": y_pred_s,
                "correct": correct
            })
            fig_tsne3 = px.scatter_3d(
                df_tsne3, x="x", y="y", z="z",
                color="label_true",
                symbol="correct",
                hover_data=["label_true", "label_pred"],
                color_discrete_map=self.color_map,
                title=f"{stage} t-SNE 3D (epoch {trainer.current_epoch})"
            )
            log_dict[f"{stage}/tsne_3d"] = fig_tsne3
    
        trainer.logger.experiment.log(log_dict)
        self.buf[stage].clear()

    # ----------------------
    # Hook 구현
    # ----------------------
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if trainer.global_rank == 0:
    #         self._collect("train", pl_module, batch)

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #     if trainer.global_rank == 0:
    #         self._collect("valid", pl_module, batch)

    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #     if trainer.global_rank == 0:
    #         self._collect("test", pl_module, batch)

    def _collect_epoch_sample(self, stage, trainer, pl_module, max_batches=4):
        # stage: "train"/"valid"/"test"
        # max_batches: 한 epoch당 몇 배치 샘플링할지
        if stage == "train":
            loader = trainer.train_dataloader
        elif stage == "valid":
            # validation_dataloaders can be list; take first
            loader = trainer.val_dataloaders
        elif stage == "test":
            loader = trainer.test_dataloaders
        else:
            return

        collected = 0
        for batch_idx, batch in enumerate(loader):
            if collected >= max_batches:
                break
            # 동일하게 _collect 내부 logic을 인라인
            if len(batch) == 4:
                signal, normal, rms, y = batch
            else:
                signal, y = batch
                normal = signal

            device = pl_module.device
            signal = signal.to(device)
            normal = normal.to(device)
            with torch.no_grad():
                _, _, cross_z, _, _ = pl_module.model(signal, normal, get_z=True)
                if getattr(pl_module, "training_mode", "recon_only") == "recon_classify":
                    _, logits, _ = pl_module.model(signal, normal, classify=True)
                    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    y_pred = np.full(len(y), -1)

            cross_z = cross_z.reshape(cross_z.size(0), -1).cpu().numpy()
            y_true = y.cpu().numpy()
            cur = list(zip(cross_z, y_true, y_pred))
            self.buf[stage].extend(cur)
            collected += 1
            if len(self.buf[stage]) > self.max_samples_per_stage:
                self.buf[stage] = self.buf[stage][-self.max_samples_per_stage:]

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._collect_epoch_sample("train", trainer, pl_module, max_batches=4)
        self._project_and_log("train", trainer, trainer.global_step)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._collect_epoch_sample("valid", trainer, pl_module, max_batches=4)
        self._project_and_log("valid", trainer, trainer.global_step)

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._collect_epoch_sample("test", trainer, pl_module, max_batches=4)
        # 테스트는 global_step 안 증가할 수 있으니 current_epoch 기준
        self._project_and_log("test", trainer, trainer.global_step)
