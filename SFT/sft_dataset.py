from torch.utils.data import Dataset
import random

class VibrationSFT_Dataset(Dataset):
    def __init__(self,
                vib_dataset
                ):
        self.vib_dataset = vib_dataset
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
        
    def __len__(self):
        return len(self.vib_dataset)

    def __getitem__(self, index):
        
        current_x, _, current_info, ref_x, _, ref_info = self.vib_dataset[index]
        current_state = current_info['merged_class']
        
        # System Prompt: 역할과 함께 '출력 형식'을 강력하게 지시
        system_prompt = (
            "You are an expert AI specializing in the diagnostics of rotating machinery based on vibration analysis. "
            "Based on the provided embeddings, classify the current condition of the equipment. "
            f"Your response MUST BE one of the following labels AND NOTHING ELSE: {', '.join(self.target_labels)}."
        )

        # User Prompt: 질문의 끝을 명사형으로 마무리하여 단답형 응답을 유도
        prompt_templates = [
            (
                "I have two vibration embeddings. The first is a baseline normal state, and the second is the current state. "
                "Please provide a diagnosis based on their difference.\n\n"
                "- Baseline (Normal) Embedding: <NORMAL_VIB_EMB>\n"
                "- Current State Embedding: <CURRENT_VIB_EMB>\n\n"
                "Diagnosis:"
            ),
            (
                "Classify the machine's fault type using the given vibration embeddings.\n\n"
                "- Normal Reference Embedding: <NORMAL_VIB_EMB>\n"
                "- Current Measurement Embedding: <CURRENT_VIB_EMB>\n\n"
                "Condition:"
            )
        ]
        user_prompt = random.choice(prompt_templates)
        
        # Assistant Prompt: 모델이 출력해야 할 정답 (단일 클래스 레이블)
        assistant_prompt = current_state
        
        return {'messages':[
                    {'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt},
                    {'role' : 'assistant', 'content' : assistant_prompt}],
                'current_x' : current_x,
                'ref_x' : ref_x
                }