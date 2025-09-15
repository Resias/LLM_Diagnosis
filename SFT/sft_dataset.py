import random
from typing import Any, Dict, List
import torch
from dataclasses import dataclass

def data_generator(vib_dataset):
    for i in range(len(vib_dataset)):
        item = vib_dataset[i]
        yield {
            "current_x": item['current_x'],
            "ref_x": item['ref_x'],
            "merged_class": item['current_info']['merged_class']
        }


def format_and_tokenize_function(example, tokenizer=None):
    """
    Return plain-text fields with no chat_template:
    - prompt_only: instructions + user cue ending with 'Diagnosis:' or 'Condition:'.
    - completion: the target label only.

    Tokenization/masking is handled by the collator.
    """
    target_labels = ['normal(healthy)', 'misalignment', 'looseness', 'unbalance', 'bearing fault']

    current_state = example['merged_class']
    if current_state == 'normal':
        current_state = 'normal(healthy)'
    elif current_state == 'bearing':
        current_state = 'bearing fault'

    system_prompt = (
        "You are an expert AI specializing in the diagnostics of rotating machinery based on vibration analysis.\n"
        "Based on the provided embeddings, classify the current condition of the equipment.\n"
        f"Your response MUST BE one of the following labels AND NOTHING ELSE: {', '.join(target_labels)}.\n"
    )

    prompt_templates = [
        (
            "I have two vibration embeddings. The first is a baseline normal state, and the second is the current state.\n"
            "Please provide a diagnosis based on their difference.\n"
            "- Baseline (Normal) Embedding: <NORMAL_VIB_EMB>\n"
            "- Current State Embedding: <CURRENT_VIB_EMB>\n"
            "Classify the current condition of the equipment:\n"
        ),
        (
            "Classify the machine's fault type using the given vibration embeddings.\n"
            "- Normal Reference Embedding: <NORMAL_VIB_EMB>\n"
            "- Current Measurement Embedding: <CURRENT_VIB_EMB>\n"
            "Classify the current condition of the equipment:\n"
        )
    ]
    user_prompt = random.choice(prompt_templates)
    assistant_prompt = current_state
    
    prompt =[
        {
            'content' : system_prompt, 'role' : 'system'
        },
        {
            'content' : user_prompt, 'role' : 'user'
        },
    ]
    completion=[
        {
            'content' : assistant_prompt, 'role' : 'assistant'
        }
    ]

    return {
        'prompt': prompt,
        'completion': completion,
        'current_x': example['current_x'],
        'ref_x': example['ref_x'],
    }

@dataclass
class VibDataCollatorWrapper:
    """
    SFTTrainer의 기본 데이터 콜레이터를 래핑하여
    'current_x', 'ref_x' 같은 커스텀 텐서를 배치에 추가합니다.
    """
    original_collator: Any  # SFTTrainer의 기본 콜레이터

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 진동 데이터만 따로 꺼내기
        current_x_list = [f.pop("current_x") for f in features]
        ref_x_list = [f.pop("ref_x") for f in features]
        
        # 2. 텍스트 관련은 원래 있었던 Collator가 처리하도록 함
        batch = self.original_collator(features)
        
        # 3. 분리해 뒀던 진동 데이터를 스택하여 배치에 추가
        try:
            batch['current_x'] = torch.stack(current_x_list)
            batch['ref_x'] = torch.stack(ref_x_list)
        except Exception:
            batch['current_x'] = torch.tensor(current_x_list)
            batch['ref_x'] = torch.tensor(ref_x_list)
        
        return batch