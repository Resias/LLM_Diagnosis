import argparse
import functools
import types
from typing import Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPooling

from peft import LoraConfig, TaskType, get_peft_model

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
    MultimodalProjectorConfig,
)
from cornstarch.models.multimodal_language_model.processing_multimodal_language_model import (
    MultimodalProcessor,
)

from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE


class FakeDataset(Dataset):
    """간단한 모의 진동-텍스트 페어를 반복 제공한다."""

    def __init__(self):
        self.vibration = torch.tensor(np.random.rand(4, 224, 224), dtype=torch.float32)
        self.prompt = "<vibration> 진동 데이터를 요약해줘."
        self.completion = " 이상 징후는 발견되지 않았습니다."
        self.text = self.prompt + self.completion

    def __len__(self):
        return 65536

    def __getitem__(self, index: int) -> dict:
        return {
            "vibration": self.vibration,
            "prompt": self.prompt,
            "completion": self.completion,
            "text": self.text,
        }


class MultimodalReasoningDataset(Dataset):
    """GRPO 샘플 생성을 위한 가짜 reasoning 데이터셋."""

    def __init__(self, length: int = 512):
        self.length = length
        self.vibration = torch.tensor(np.random.rand(4, 224, 224), dtype=torch.float32)
        # 데모용 GT: 간단히 키워드(예: "이상")를 기대값으로 둔다.
        self.gt_token = "이상"

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> dict:
        prompt = "<vibration> 진동 데이터를 분석하고 가능한 이상 징후를 설명해줘."
        # 예시 GT: 답변에 "이상" 키워드가 포함되어야 한다고 가정
        gt = self.gt_token
        return {"prompt": prompt, "vibration": self.vibration, "gt": gt}


def collate_fn(batches: list[dict], processor: MultimodalProcessor) -> dict:
    """Cornstarch 프로세서를 활용해 모달리티별 입력을 LLM 형식으로 정리한다."""
    vibrations = [batch["vibration"] for batch in batches]
    prompts = [batch.get("prompt", "") for batch in batches]
    completions = [batch.get("completion", "") for batch in batches]
    texts = [p + c for p, c in zip(prompts, completions)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed = processor(
        encoder_inputs={"vibration": {"vibration": vibrations}},
        llm_inputs={"text": texts, "padding": True},
        return_tensors="pt",
    )

    tokenizer = processor.llm_tokenizer
    max_length = processed["input_ids"].shape[1]
    offsets = None
    if getattr(tokenizer, "is_fast", False):
        offsets = tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )["offset_mapping"]

    labels = processed["input_ids"].clone()
    attention_mask = processed["attention_mask"]
    labels[attention_mask == 0] = -100

    if offsets is not None:
        prompt_char_lens = [len(p) for p in prompts]
        for i, prompt_char_len in enumerate(prompt_char_lens):
            if prompt_char_len <= 0:
                continue
            token_offsets = offsets[i]
            prompt_mask = token_offsets[:, 1] <= prompt_char_len
            labels[i][prompt_mask] = -100

    inputs: dict[str, torch.Tensor] = {}
    for key, value in processed.items():
        if torch.is_tensor(value):
            value = value.to(device=device)
            # 진동 텐서는 원본 dtype을 유지해야 VisionEncoder가 동작한다.
            if value.is_floating_point() and key != "vibration":
                value = value.to(dtype=torch.bfloat16)
            inputs[key] = value
        else:
            inputs[key] = value

    inputs["labels"] = labels.to(device=device)
    inputs["prompts"] = prompts
    inputs["completions"] = completions

    for value in inputs.values():
        if torch.is_tensor(value) and value.is_floating_point():
            value.requires_grad_(True)

    return inputs


class VibrationEncoderConfig(PretrainedConfig):
    model_type = "vibration_encoder"

    def __init__(
        self,
        input_channels: int = 2,
        input_length: int = 1024,
        hidden_size: int = 256,
        add_pooling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.add_pooling = add_pooling


class VibrationEncoderHF(PreTrainedModel):
    """사용자 정의 진동 인코더를 HF `PreTrainedModel` 인터페이스로 감싸는 래퍼."""

    config_class = VibrationEncoderConfig
    main_input_name = "vibration"

    def __init__(self, config: VibrationEncoderConfig, vib_ae: Optional[nn.Module] = None):
        super().__init__(config)
        if vib_ae is None:
            raise ValueError(
                "VibrationEncoderHF requires a `vib_ae` (nn.Module with `.encode(...)`). "
                "Pass it via ctor or from_pretrained(..., vib_ae=...)."
            )
        self.vib_encoder = vib_ae

    def forward(
        self,
        vibration: torch.FloatTensor,
        return_dict: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.vib_encoder.encode(vibration)
        if isinstance(z, (tuple, list)):
            z = z[0]

        if z.dim() == 2:
            last_hidden_state = z.unsqueeze(1)
            pooler_output = z
        elif z.dim() == 3:
            last_hidden_state = z
            pooler_output = z.mean(dim=1)
        else:
            raise ValueError(f"Unexpected z shape {z.shape}. Expect (B, D) or (B, T, D).")

        if not return_dict:
            return last_hidden_state, pooler_output

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=None if not output_hidden_states else (last_hidden_state,),
            attentions=None,
        )


def build_vibration_projector(embedding_dim: int, token_embed_dim: int) -> MultimodalProjector:
    """진동 인코더 출력을 LLM 토큰 임베딩 차원으로 사상하는 프로젝터."""
    projection = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=embedding_dim, out_features=int(embedding_dim * 2)),
        nn.Sigmoid(),
        nn.Linear(in_features=int(embedding_dim * 2), out_features=token_embed_dim),
    )

    proj_cfg = MultimodalProjectorConfig(
        in_features=embedding_dim,
        out_features=token_embed_dim,
    )
    return MultimodalProjector(config=proj_cfg, projection=projection)


class VibrationProcessor:
    """Cornstarch 프로세서에서 요구하는 배치 텐서 형태로 진동 데이터를 정리한다."""

    def __call__(
        self,
        vibration: list[torch.Tensor] | torch.Tensor,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        if isinstance(vibration, list):
            vibration = [torch.as_tensor(v) for v in vibration]
            vibration = torch.stack(vibration, dim=0)
        else:
            vibration = torch.as_tensor(vibration)
            if vibration.ndim == 2:
                vibration = vibration.unsqueeze(0)

        vibration = vibration.to(dtype=torch.float32)

        if return_tensors != "pt":
            raise ValueError(f"Unsupported tensor type: {return_tensors}")

        return BatchFeature(data={"vibration": vibration})


def vibration_num_features(inputs: dict, outputs: BatchFeature) -> list[int]:
    """모달 토큰 개수를 LLM에 알려주기 위한 헬퍼."""
    batch_size = outputs["vibration"].shape[0]
    return [1] * batch_size


def ensure_tokenizer(tokenizer: AutoTokenizer) -> bool:
    """필수 특수 토큰을 추가하고 실제로 vocab이 확장됐는지 여부를 반환한다."""
    resized = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        resized = True
    added = tokenizer.add_special_tokens({"additional_special_tokens": ["<vibration>"]})
    return resized or added > 0


def patch_llm_forward(llm: PreTrainedModel) -> None:
    """Cornstarch가 넘겨주는 불필요한 인자를 제거해 Qwen3와의 충돌을 방지한다."""
    original_forward = llm.forward

    def forward_without_extra_kwargs(self, *args, **kwargs):
        kwargs.pop("position_embeddings", None)
        kwargs.pop("hidden_states", None)
        return original_forward(*args, **kwargs)

    llm.forward = types.MethodType(forward_without_extra_kwargs, llm)


def build_vibration_module(llm_hidden_size: int) -> ModalEncoderModule:
    embedding_dim = 768
    pretrained_ae = VisionTransformerAE(num_classes=5)
    vib_cfg = VibrationEncoderConfig(
        input_channels=2,
        input_length=224,
        hidden_size=embedding_dim,
    )
    vibration_encoder_hf = VibrationEncoderHF(vib_cfg, vib_ae=pretrained_ae)
    projector = build_vibration_projector(
        embedding_dim=embedding_dim,
        token_embed_dim=llm_hidden_size,
    )
    return ModalEncoderModule(
        model=vibration_encoder_hf,
        projector=projector,
    )


def build_multimodal_system(
    apply_lora: bool = True,
    cache_dir: str = "/workspace/llm_cache",
) -> tuple[MultimodalModel, AutoTokenizer, MultimodalProcessor]:
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        device_map="auto",
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        cache_dir=cache_dir,
    )

    if ensure_tokenizer(tokenizer):
        llm.resize_token_embeddings(len(tokenizer))

    vibration_module = build_vibration_module(llm.config.hidden_size)

    if apply_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        llm = get_peft_model(llm, peft_config)

    patch_llm_forward(llm)

    mllm = MultimodalModel(
        encoders={"vibration": vibration_module},
        language_model=llm,
    )

    # LLM 임베딩 디바이스와 인코더 디바이스를 일치시켜 디바이스 mismatch 방지
    try:
        lm_device = llm.get_input_embeddings().weight.device
    except Exception:
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mllm.encoders["vibration"].module.to(lm_device)
    mllm.encoders["vibration"].projector.to(lm_device)

    vibration_processor = VibrationProcessor()
    mm_processor = MultimodalProcessor(
        encoder_processors={"vibration": vibration_processor},
        llm_tokenizer=tokenizer,
        num_feature_calculation_funcs={"vibration": vibration_num_features},
        model=mllm,
    )
    llm.resize_token_embeddings(len(tokenizer))
    return mllm, tokenizer, mm_processor


class MultimodalGRPOCollator:
    """GRPO 학습에 사용할 멀티모달 collator."""

    def __init__(self, processor: MultimodalProcessor, device: torch.device | None = None):
        self.processor = processor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, features: list[dict]) -> dict:
        vibrations = [item["vibration"] for item in features]
        prompts = [item["prompt"] for item in features]
        gts = [item.get("gt") for item in features]

        processed = self.processor(
            encoder_inputs={"vibration": {"vibration": vibrations}},
            llm_inputs={"text": prompts, "padding": True, "truncation": True},
            return_tensors="pt",
        )

        batch: dict[str, torch.Tensor | list[str]] = {}
        for key, value in processed.items():
            if torch.is_tensor(value):
                value = value.to(device=self.device)
                if value.is_floating_point() and key != "vibration":
                    value = value.to(dtype=torch.bfloat16)
            batch[key] = value

        batch["prompts"] = prompts
        batch["gts"] = gts
        return batch


class CornstarchPolicyConfig(PretrainedConfig):
    model_type = "cornstarch-policy"


class BaseTrainer(pl.LightningModule):
    """Cornstarch 멀티모달 모델 학습을 위한 공통 Lightning 베이스 모듈."""

    def __init__(
        self,
        *,
        apply_lora: bool = True,
        lr: float = 1e-3,
        encoder_mode: Optional[dict[str, tuple[bool, bool]]] = None,
        llm_mode: bool = True,
    ):
        super().__init__()
        self.model, self.tokenizer, self.processor = build_multimodal_system(apply_lora=apply_lora)
        self.lr = lr
        self.encoder_mode = encoder_mode or {"vibration": (False, True)}
        self.llm_mode = llm_mode
        self._align_modal_modules()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _align_modal_modules(self) -> None:
        """LLM과 진동 인코더/프로젝터의 디바이스를 일치시킨다."""
        try:
            lm_device = self.model.language_model.get_input_embeddings().weight.device
        except Exception:
            lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, "encoders") and "vibration" in self.model.encoders:
            self.model.encoders["vibration"].module.to(lm_device)
            self.model.encoders["vibration"].projector.to(lm_device)
        self._modal_device = lm_device

    def on_fit_start(self) -> None:
        self._align_modal_modules()
        self.model.train(
            encoders_mode=self.encoder_mode,
            llm_mode=self.llm_mode,
        )

    def _create_optimizer(self) -> Adam:
        return Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def configure_optimizers(self):
        return self._create_optimizer()


class SFTTrainer(BaseTrainer):
    """감독학습(SFT)을 위한 Lightning 모듈."""

    def __init__(
        self,
        total_steps: int = 10,
        *,
        lr: float = 1e-3,
        warmup_ratio: float = 0.1,
        apply_lora: bool = True,
    ):
        super().__init__(apply_lora=apply_lora, lr=lr, llm_mode=False)
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio

    def training_step(self, batch, batch_idx: int):
        forward_inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        outputs = self.model(**forward_inputs)
        loss = outputs.loss
        batch_size = batch["input_ids"].size(0) if torch.is_tensor(batch.get("input_ids")) else None
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = self._create_optimizer()
        num_training_steps = max(1, self.total_steps)
        num_warmup_steps = max(0, int(num_training_steps * self.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class GRPOTrainer(BaseTrainer):
    """Cornstarch MultimodalModel용 GRPO 학습 Lightning 모듈."""

    def __init__(
        self,
        reward_fns: list[Callable[[list[str], list[str], list[object]], list[float]]],
        *,
        reward_weights: Optional[list[float]] = None,
        num_generations: int = 2,
        max_completion_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        lr: float = 1e-6,
        apply_lora: bool = True,
    ):
        super().__init__(apply_lora=apply_lora, lr=lr, llm_mode=True)
        self.reward_fns = reward_fns
        self.reward_weights = reward_weights
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.automatic_optimization = False

    @torch.no_grad()
    def _generate(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """프롬프트 배치로부터 completions를 생성하고 반환."""
        device = self._modal_device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = attention_mask.sum(dim=1)

        gen_kwargs = dict(
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=self.num_generations,
            max_new_tokens=self.max_completion_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "vibration")}
        for key, value in gen_batch.items():
            if torch.is_tensor(value):
                gen_batch[key] = value.to(device)

        sequences = self.model.generate(**gen_batch, **gen_kwargs)
        return sequences, prompt_lens, attention_mask

    def _compute_logps_for_completions(
        self,
        sequences: torch.Tensor,
        prompt_lens: torch.Tensor,
        vib: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """completion 토큰에 대한 (per-token) logp와 마스크를 계산."""
        device = self._modal_device
        sequences = sequences.to(device)
        batch_size_gen = sequences.size(0)

        vib_rep = vib.to(device)
        if vib_rep.size(0) * self.num_generations == batch_size_gen:
            vib_rep = vib_rep.repeat_interleave(self.num_generations, dim=0)
        elif vib_rep.size(0) != batch_size_gen:
            raise ValueError("vibration batch size mismatch with generated sequences")

        attn = (sequences != self.tokenizer.pad_token_id).long()
        inputs = {
            "input_ids": sequences,
            "attention_mask": attn,
            "vibration": vib_rep,
            "labels": sequences.clone(),
        }

        outputs = self.model(**inputs)
        logits = outputs.logits
        logprobs = logits.log_softmax(dim=-1)

        next_token_ids = sequences[:, 1:]
        token_logps = torch.gather(
            logprobs[:, :-1, :],
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        prompt_lens_rep = prompt_lens.repeat_interleave(self.num_generations)
        L_total = sequences.size(1)
        idx = torch.arange(L_total - 1, device=device).unsqueeze(0).expand_as(token_logps)
        comp_mask = (idx >= (prompt_lens_rep.unsqueeze(1) - 1)) & (next_token_ids != self.tokenizer.pad_token_id)
        comp_mask = comp_mask.long()
        return token_logps, comp_mask

    def _decode_completions(self, sequences: torch.Tensor, prompt_lens: torch.Tensor) -> list[str]:
        seqs = sequences.cpu()
        plens = prompt_lens.cpu()
        completions: list[str] = []
        B = plens.size(0)
        for i in range(B):
            for g in range(self.num_generations):
                idx = i * self.num_generations + g
                comp_ids = seqs[idx, plens[i] :]
                text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                completions.append(text)
        return completions

    def training_step(self, batch, batch_idx: int):
        prompts = batch.get("prompts") or []
        gts = batch.get("gts") or []
        vib = batch.get("vibration")
        opt = self.optimizers()

        with torch.no_grad():
            sequences, prompt_lens, _ = self._generate(batch)

        completions = self._decode_completions(sequences, prompt_lens)
        B = len(prompts)

        per_fn_rewards: list[torch.Tensor] = []
        prompts_rep = prompts * self.num_generations
        if isinstance(gts, list):
            gts_rep = gts * self.num_generations
        else:
            gts_rep = [gts] * (len(prompts) * self.num_generations)
        for fn in self.reward_fns:
            rewards = fn(prompts_rep, completions, gts_rep)
            per_fn_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self._modal_device))

        if self.reward_weights is None:
            weights = [1.0 / len(per_fn_rewards)] * len(per_fn_rewards) if per_fn_rewards else []
        else:
            assert len(self.reward_weights) == len(per_fn_rewards), "reward_weights 길이가 reward_fns와 같아야 합니다."
            weights = self.reward_weights

        rewards = sum(w * r for w, r in zip(weights, per_fn_rewards)) if per_fn_rewards else torch.zeros(B * self.num_generations, device=self._modal_device)
        rewards = rewards.view(B, self.num_generations)
        group_mean = rewards.mean(dim=1, keepdim=True)
        advantages = (rewards - group_mean).view(-1)

        token_logps, comp_mask = self._compute_logps_for_completions(sequences, prompt_lens.to(self._modal_device), vib)
        comp_lengths = comp_mask.sum(dim=1).clamp_min(1)
        seq_logp = (token_logps * comp_mask).sum(dim=1) / comp_lengths

        loss = -(seq_logp * advantages).mean()

        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()

        batch_size = B if B else len(seq_logp)
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        self.log("avg_reward", rewards.mean(), on_step=True, prog_bar=False, batch_size=batch_size)
        return loss


def run_supervised_training(
    total_steps: int = 10,
    batch_size: int = 4,
):
    """Lightning 기반 SFT 학습 루프."""
    module = SFTTrainer(total_steps=total_steps)
    dataset = FakeDataset()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=functools.partial(collate_fn, processor=module.processor),
    )

    trainer = pl.Trainer(
        max_steps=total_steps,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=dataloader)


def run_grpo_training(
    num_epochs: int = 1,
    grpo_batch_size: int = 4,
    max_prompt_length: int = 512,
    max_completion_length: int = 128,
):
    """Lightning 기반 GRPO 학습 루프."""
    # 데이터/콜레이터 준비
    train_dataset = MultimodalReasoningDataset()

    # 예시 리워드 1: 길이 기반 보상 (길수록 가산). gts는 사용하지 않음
    def reward_length(prompts: list[str], completions: list[str], gts: list[object]) -> list[float]:
        return [float(len(c)) for c in completions]

    # 예시 리워드 2: GT 기반 보상 (정답 키워드가 completion에 포함되면 +1)
    def make_gt_keyword_reward() -> Callable[[list[str], list[str], list[object]], list[float]]:
        def _fn(prompts: list[str], completions: list[str], gts: list[object]) -> list[float]:
            out = []
            for c, gt in zip(completions, gts):
                key = str(gt) if gt is not None else ""
                out.append(1.0 if key in (c or "") else 0.0)
            return out
        return _fn

    reward_fns = [reward_length, make_gt_keyword_reward()]
    reward_weights = [0.5, 0.5]

    module = GRPOTrainer(
        reward_fns=reward_fns,
        reward_weights=reward_weights,
        num_generations=2,
        max_completion_length=max_completion_length,
        temperature=1.0,
        top_p=1.0,
        lr=1e-6,
    )

    data_collator = MultimodalGRPOCollator(processor=module.processor)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=grpo_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=data_collator,
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=dataloader)


def main():
    parser = argparse.ArgumentParser(description="Cornstarch 멀티모달 + TRL GRPO 통합 예제")
    parser.add_argument(
        "--trainer",
        choices=("supervised", "grpo"),
        default="grpo",
        help="학습 방식을 선택합니다. grpo는 커스텀 GRPO 루프 사용",
    )
    parser.add_argument("--steps", type=int, default=10, help="감독학습 반복 횟수")
    parser.add_argument("--epochs", type=int, default=1, help="GRPO 학습 epoch 수")
    parser.add_argument("--grpo_batch", type=int, default=4, help="GRPO 배치 크기")
    args = parser.parse_args()

    if args.trainer == "supervised":
        run_supervised_training(total_steps=args.steps)
    else:
        run_grpo_training(
            num_epochs=args.epochs,
            grpo_batch_size=args.grpo_batch,
        )


if __name__ == "__main__":
    main()
