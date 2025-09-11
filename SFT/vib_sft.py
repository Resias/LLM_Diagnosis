import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
from typing import Dict, Any, Union, Optional, List, Tuple
from itertools import chain



class VibrationSFTTrainer(SFTTrainer):
    """
    SFTTrainer를 상속받아 진동 임베딩을 처리하는 커스텀 트레이너.

    1. __init__: vib_tokenizer를 추가 인자로 받아 초기화합니다.
    2. compute_loss: 데이터셋의 'current_x', 'ref_x'를 vib_tokenizer로 임베딩하고,
                     텍스트 임베딩의 특수 토큰 위치에 해당 임베딩을 삽입하여 손실을 계산합니다.
    """
    def __init__(self, *args, vib_tokenizer: Optional[nn.Module] = None, **kwargs):
        # 1. 기존 SFTTrainer의 __init__ 함수를 그대로 호출합니다.
        super().__init__(*args, **kwargs)

        # vib_tokenizer가 제공되지 않으면 에러를 발생시킵니다.
        if vib_tokenizer is None:
            raise ValueError("VibrationSFTTrainer requires a `vib_tokenizer`.")
        self.vib_tokenizer = vib_tokenizer
        
        # 특수 토큰의 ID를 미리 찾아둡니다. 이 토큰들은 language tokenizer에 미리 추가되어야 합니다.
        self.normal_vib_token_id = self.tokenizer.convert_tokens_to_ids("<NORMAL_VIB_EMB>")
        self.current_vib_token_id = self.tokenizer.convert_tokens_to_ids("<CURRENT_VIB_EMB>")
        
        # 특수 토큰이 vocabulary에 없는 경우 에러 처리
        if self.tokenizer.unk_token_id in [self.normal_vib_token_id, self.current_vib_token_id]:
             raise ValueError(
                "Special tokens '<NORMAL_VIB_EMB>' or '<CURRENT_VIB_EMB>' are not found in the tokenizer's vocabulary. "
                "Please add them using `tokenizer.add_special_tokens` before initializing the trainer."
            )

        # ------------------- 추가된 부분 -------------------
    def create_optimizer(self):
        """
        model과 vib_tokenizer의 파라미터를 모두 학습하는 옵티마이저를 생성합니다.
        """
        # TrainingArguments에서 옵티마이저 하이퍼파라미터를 가져옵니다.
        optimizer_kwargs = {
            "lr": self.args.learning_rate,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        # model과 vib_tokenizer의 파라미터를 하나로 합칩니다.
        combined_parameters = chain(self.model.parameters(), self.vib_tokenizer.alignment_layer.parameters())
        
        # 합쳐진 파라미터 그룹으로 AdamW 옵티마이저를 생성합니다.
        self.optimizer = AdamW(combined_parameters, **optimizer_kwargs)
        
        # 생성된 옵티마이저를 반환합니다. 스케줄러는 Trainer가 알아서 처리합니다.
        return self.optimizer

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        입력 배치를 받아 임베딩을 교체하고 손실을 계산하는 핵심 메소드.
        """
        # Dataset에서 온 진동 텐서('current_x', 'ref_x')를 배치에서 분리합니다.
        # .pop()을 사용해 이후 모델에 직접 전달되지 않도록 합니다.
        current_x = inputs.pop("current_x").to(self.args.device)
        ref_x = inputs.pop("ref_x").to(self.args.device)

        # 2. vib_tokenizer를 사용해 진동 텐서를 임베딩으로 변환합니다.
        # self.vib_tokenizer가 GPU 연산을 지원하도록 .to(device) 처리가 되어있어야 합니다.
        current_emb = self.vib_tokenizer(current_x) # shape: (batch_size, embedding_dim)
        ref_emb = self.vib_tokenizer(ref_x)         # shape: (batch_size, embedding_dim)

        # `input_ids`를 모델의 워드 임베딩 레이어를 통과시켜 텍스트 임베딩을 얻습니다.
        input_ids = inputs.pop("input_ids")
        text_embeds = model.get_input_embeddings()(input_ids)

        # 배치 내 각 샘플에 대해 특수 토큰 위치를 찾아 진동 임베딩으로 교체합니다.
        for i in range(input_ids.shape[0]):
            # <NORMAL_VIB_EMB> 토큰의 위치를 찾습니다.
            normal_token_indices = (input_ids[i] == self.normal_vib_token_id).nonzero(as_tuple=True)[0]
            if normal_token_indices.nelement() > 0:
                # 해당 위치의 텍스트 임베딩을 ref_emb(정상 상태 진동 임베딩)로 교체합니다.
                text_embeds[i, normal_token_indices[0], :] = ref_emb[i]

            # <CURRENT_VIB_EMB> 토큰의 위치를 찾습니다.
            current_token_indices = (input_ids[i] == self.current_vib_token_id).nonzero(as_tuple=True)[0]
            if current_token_indices.nelement() > 0:
                # 해당 위치의 텍스트 임베딩을 current_emb(현재 상태 진동 임베딩)로 교체합니다.
                text_embeds[i, current_token_indices[0], :] = current_emb[i]
        
        # 수정된 임베딩을 모델 입력으로 설정합니다.
        # `input_ids`는 더 이상 필요 없으므로 `inputs_embeds`를 대신 사용합니다.
        inputs["inputs_embeds"] = text_embeds
        
        # 이제 상위 클래스(Trainer)의 compute_loss 로직을 호출하여 손실을 계산합니다.
        # Trainer.compute_loss는 내부적으로 model(**inputs)를 호출하므로,
        # 우리가 수정한 inputs_embeds가 모델에 전달됩니다.
        return super().compute_loss(model, inputs, return_outputs)