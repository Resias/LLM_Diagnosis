import torch

from torch.utils.data import Dataset
from data.order_dataset import ExtendedOrderFreqDataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from trl import GRPOTrainer, GRPOConfig
from utils.reward import reward_format, reward_accuracy


# Dataset
class VibrationSFTDatasetEnglish(Dataset):
    def __init__(self, vibration_dataset):
        self.vibration_dataset = vibration_dataset

    def __len__(self):
        return len(self.vibration_dataset)

    def __getitem__(self, idx):
        (
            description_sample, description_normal, class_name
        ) = self.vibration_dataset[idx]

        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer." 
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively," 
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        )

        user_prompt = (
            "Act as an expert in vibration-based diagnosis of rotating machinery. "
            "Analyze the order frequency description of both the normal and current vibration signals I provide, "
            "and accurately diagnose the current condition of the equipment. "
            "Possible diagnostic classes are: looseness, normal, unbalance, misalignment, bearing."
            
            f"Normal state (x-axis): {description_normal[0]}\n"
            f"Normal state (y-axis): {description_normal[1]}\n"
            f"Current state (x-axis): {description_sample[0]}\n"
            f"Current state (y-axis): {description_sample[1]}\n"
            "Diagnose condition: looseness, normal, unbalance, misalignment, bearing."
        )

        # Generate a detailed CoT reasoning string emphasizing <VIB_EMB>
        assistant_response = f"The diagnosis result is {class_name}."

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        return {
            "prompt": prompt_only,
            "answers": assistant_response
        }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vibration_dataset = ExtendedOrderFreqDataset(data_root='/workspace/dataset')

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    llm.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)


    dataset = VibrationSFTDatasetEnglish(tokenizer, vibration_dataset)
    
    # 3. Configure training
    training_args = GRPOConfig(
        output_dir="output"
    )

    # 4. Initialize and train
    trainer = GRPOTrainer(
        model=llm,  
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward_format, reward_accuracy],
    )
    trainer.train()