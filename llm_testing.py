import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.unet import UnetVAE

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

class Projector(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.projector(x)

# 3. Full Model wrapping
class FullModel(nn.Module):
    def __init__(self, u_encoder, projector, llm, vib_token_id):
        super().__init__()
        self.u_encoder = u_encoder
        self.projector = projector
        self.llm = llm
        self.vib_token_id = vib_token_id

    def forward(self, input_ids, attention_mask, vibration_signal, labels=None):
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        u_features = self.u_encoder(vibration_signal)
        vib_embed = self.projector(u_features).unsqueeze(1)
        mask = (input_ids == self.vib_token_id).unsqueeze(-1)
        input_embeds = torch.where(mask, vib_embed, input_embeds)
        outputs = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

class VibrationSFTDatasetEnglish(Dataset):
    def __init__(self, tokenizer, vibration_dataset, max_length=2048):
        """
        vibration_dataset : list of tuples
            (signal_tensor, signal_info, ref_tensor, ref_info, freq_feature_dict, time_feature_dict, ref_freq_feature_dict, ref_time_feature_dict)
        """
        self.vibration_dataset = vibration_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vib_token_id = tokenizer.convert_tokens_to_ids('<VIB_EMB>')

    def __len__(self):
        return len(self.vibration_dataset)

    def __getitem__(self, idx):
        (
            signal_tensor, signal_info, ref_tensor, ref_info,
            freq_feature_dict, time_feature_dict,
            ref_freq_feature_dict, ref_time_feature_dict
        ) = self.vibration_dataset[idx]

        # System prompt (role)
        system_prompt = (
            "Act as an expert in vibration-based diagnosis of rotating machinery.\n"
            "Analyze the statistical features of both the reference (normal) and current vibration signals I provide, "
            "and accurately diagnose the current condition of the equipment.\n"
            "Possible diagnostic classes are: looseness, normal, unbalance, misalignment, bearing."
        )

        # User prompt (feature injection)
        user_prompt = f"""
            Machine operating information:
            - Rotational speed: {signal_info['rpm']} RPM
            - Load condition: {signal_info['load_condition'] if signal_info['load_condition'] != 'unknown' else 'unknown'}

            Diagnostic target classes: looseness, normal, unbalance, misalignment, bearing.

            Current vibration state: <VIB_EMB>

            Time domain features (reference, x-axis): {ref_time_feature_dict[0]}
            Time domain features (reference, y-axis): {ref_time_feature_dict[1]}
            Frequency domain features (reference, x-axis): {ref_freq_feature_dict[0]}
            Frequency domain features (reference, y-axis): {ref_freq_feature_dict[1]}

            Time domain features (current, x-axis): {time_feature_dict[0]}
            Time domain features (current, y-axis): {time_feature_dict[1]}
            Frequency domain features (current, x-axis): {freq_feature_dict[0]}
            Frequency domain features (current, y-axis): {freq_feature_dict[1]}

            Based on the provided information, analyze and determine the current condition of the machine from the possible diagnostic classes.
        """

        # Assistant response (supervised label)
        assistant_response = f"The diagnosis result is: {signal_info['class_name']}."

        # Full conversation format
        full_prompt = (
            f"System: {system_prompt}\n"
            f"User: {user_prompt}\n"
            f"Assistant: {assistant_response}"
        )

        # Tokenize full prompt
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()

        # Mask prompt part for SFT
        prompt_only = (
            f"System: {system_prompt}\n"
            f"User: {user_prompt}\n"
            f"Assistant:"
        )
        prompt_len = len(self.tokenizer(prompt_only, return_tensors='pt')['input_ids'][0])
        labels[:prompt_len] = -100  # only calculate loss on response part

        return {
            "input_ids": input_ids,
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "labels": labels,
            "signal_tensor": signal_tensor,  # vibration input for embedding injection
            "ref_tensor": ref_tensor
        }

if __name__ == '__main__':

    
    unet_vae = UnetVAE()
    
    
    
    
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': ['<VIB_EMB>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    llm.resize_token_embeddings(len(tokenizer))

    # PEFT (LoRA 적용)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.05
    )
    llm = get_peft_model(llm, peft_config)
    llm.print_trainable_parameters()





# 5. Data Collator (vibration_signal 포함)
def data_collator(features):
    input_ids = torch.stack([f['input_ids'] for f in features])
    attention_mask = torch.stack([f['attention_mask'] for f in features])
    labels = torch.stack([f['labels'] for f in features])
    vibration_signal = torch.stack([f['vibration_signal'] for f in features])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "vibration_signal": vibration_signal
    }

# 6. 전체 Trainer 호환 모델 포장
class TrainerModelWrapper(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model

    def forward(self, input_ids, attention_mask, labels, vibration_signal):
        return self.model(input_ids, attention_mask, vibration_signal, labels=labels)

# 7. 전체 학습 실행
u_encoder = UEncoder().to("cuda")
projector = Projector(output_dim=llm.config.hidden_size).to("cuda")
vib_token_id = tokenizer.convert_tokens_to_ids('<VIB_EMB>')
full_model = FullModel(u_encoder, projector, llm, vib_token_id).to("cuda")

dataset = VibrationDataset(tokenizer)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    fp16=True,
    save_steps=10,
    save_total_limit=2,
    logging_steps=1,
    report_to="none"
)

trainer = Trainer(
    model=TrainerModelWrapper(full_model),
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()