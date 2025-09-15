import torch
import torch.nn as nn
import numpy as np
from trl import SFTTrainer
from typing import Dict, Any, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

TARGET_LABELS = ['normal(healthy)', 'misalignment', 'looseness', 'unbalance', 'bearing fault']

class VibrationSFTTrainer(SFTTrainer):
    """
    SFTTrainer를 상속받아 진동 임베딩을 처리하는 커스텀 트레이너.

    1. __init__: vib_tokenizer를 추가 인자로 받아 초기화합니다.
    2. compute_loss: 데이터셋의 'current_x', 'ref_x'를 vib_tokenizer로 임베딩하고,
                     텍스트 임베딩의 특수 토큰 위치에 해당 임베딩을 삽입하여 손실을 계산합니다.
    """
    def __init__(self,
                    model,
                    args=None,
                    data_collator=None,
                    train_dataset=None,
                    eval_dataset=None,
                    processing_class=None,
                    compute_loss_func=None,
                    compute_metrics=None,
                    callbacks=None,
                    optimizers=None,
                    optimizer_cls_and_kwargs=None,
                    preprocess_logits_for_metrics=None,
                    peft_config=None,
                    formatting_func=None,
                    vib_tokenizer: Optional[nn.Module] = None):
        # 1. 기존 SFTTrainer의 __init__ 함수를 그대로 호출합니다.
        super().__init__(
                        model,
                        args,
                        data_collator,
                        train_dataset,
                        eval_dataset,
                        processing_class,
                        compute_loss_func,
                        compute_metrics,
                        callbacks,
                        optimizers,
                        optimizer_cls_and_kwargs,
                        preprocess_logits_for_metrics,
                        peft_config,
                        formatting_func)

        # vib_tokenizer가 제공되지 않으면 에러를 발생시킵니다.
        if vib_tokenizer is None:
            raise ValueError("VibrationSFTTrainer requires a `vib_tokenizer`.")
        device = model.device
        self.vib_tokenizer = vib_tokenizer.to(device=device)
        
        # 특수 토큰의 ID를 미리 찾아둡니다. 이 토큰들은 language tokenizer에 미리 추가되어야 합니다.
        self.normal_vib_token_id = self.tokenizer.convert_tokens_to_ids("<NORMAL_VIB_EMB>")
        self.current_vib_token_id = self.tokenizer.convert_tokens_to_ids("<CURRENT_VIB_EMB>")
        
        # 특수 토큰이 vocabulary에 없는 경우 에러 처리
        if self.tokenizer.unk_token_id in [self.normal_vib_token_id, self.current_vib_token_id]:
             raise ValueError(
                "Special tokens '<NORMAL_VIB_EMB>' or '<CURRENT_VIB_EMB>' are not found in the tokenizer's vocabulary. "
                "Please add them using `tokenizer.add_special_tokens` before initializing the trainer."
             )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        입력 배치를 받아 임베딩을 교체하고 손실을 계산하는 핵심 메소드.
        """


        current_x = inputs.pop("current_x").to(self.args.device)
        ref_x = inputs.pop("ref_x").to(self.args.device)


        current_emb = self.vib_tokenizer(current_x) # shape: (batch_size, embedding_dim)
        ref_emb = self.vib_tokenizer(ref_x)         # shape: (batch_size, embedding_dim)


        input_ids = inputs.pop("input_ids")
        text_embeds = model.get_input_embeddings()(input_ids)

        # dtype 일치 (일반적으로 bfloat16/float16)
        current_emb = current_emb.to(dtype=text_embeds.dtype)
        ref_emb = ref_emb.to(dtype=text_embeds.dtype)

        # 특수 토큰 위치 찾고 진동 임베딩으로 교체
        for i in range(input_ids.shape[0]):
            # <NORMAL_VIB_EMB> 토큰의 위치를 찾습니다.
            normal_token_indices = (input_ids[i] == self.normal_vib_token_id).nonzero(as_tuple=True)[0]
            if normal_token_indices.nelement() > 0:
                text_embeds[i, normal_token_indices[0], :] = ref_emb[i]

            # <CURRENT_VIB_EMB> 토큰의 위치를 찾습니다.
            current_token_indices = (input_ids[i] == self.current_vib_token_id).nonzero(as_tuple=True)[0]
            if current_token_indices.nelement() > 0:
                text_embeds[i, current_token_indices[0], :] = current_emb[i]

        # 모델에 input_inds가 아니라 input_embeds를 전달해야하기 때문에 input_ids 제거
        inputs["inputs_embeds"] = text_embeds
        if "input_ids" in inputs:
            inputs.pop("input_ids") 
        
        # labels는 모델 호출 전에 분리하여 내부 loss 계산을 비활성화합니다.
        labels = inputs.pop("labels").to(self.args.device)

        # 모델 호출 (logits만 필요)
        outputs = model(**inputs)

        loss = self.label_smoother(outputs, labels, shift_labels=True)

        return (loss, outputs) if return_outputs else loss

        


def extract_label_from_response(text):
    for label in TARGET_LABELS:
        if label in text:
            return label
    return "unknown" # 레이블을 찾지 못한 경우

def compute_metrics(p: EvalPrediction, tokenizer):
    # p.predictions는 로짓(logits) 텐플, 보통 (logits, past_key_values) 형태
    # 첫 번째 요소인 logits를 사용
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # 가장 높은 로짓을 가진 토큰의 인덱스를 예측값으로 선택
    preds_ids = np.argmax(logits, axis=-1)
    
    # p.label_ids는 실제 레이블 토큰 ID
    # 패딩 토큰(-100)은 계산에서 제외하기 위해 tokenizer.pad_token_id로 변경
    labels_ids = np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id)

    # 토큰 ID를 실제 텍스트로 디코딩
    preds_str = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # 디코딩된 텍스트에서 실제 레이블만 추출
    preds_cleaned = [extract_label_from_response(s) for s in preds_str]
    labels_cleaned = [extract_label_from_response(s) for s in labels_str]
    
    # scikit-learn을 사용하여 성능 지표 계산
    # average='weighted'는 클래스 불균형을 고려하여 F1 score 등을 계산
    precision, recall, f_beta, _ = precision_recall_fscore_support(labels_cleaned, preds_cleaned, labels=TARGET_LABELS, average='macro', zero_division=)
    acc = accuracy_score(labels_cleaned, preds_cleaned)
    
    # 결과를 딕셔너리 형태로 반환
    return {
        'accuracy': acc,
        'f1': f_beta,
        'precision': precision,
        'recall': recall
    }
    
def preprocess_logits_for_metrics(logits, labels):
    """
    logits 텐서에서 argmax를 취해 가장 가능성 높은 토큰 ID를 반환합니다.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)