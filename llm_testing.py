import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data.dataset import VibrationDataset
from models.unet import UnetVAE

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig


class Projector(nn.Module):
    def __init__(self, in_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, output_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.projector(x)

class FullModel(nn.Module):
    def __init__(self, encoder, projector, llm, vib_token_id):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.vib_token_id = vib_token_id
        
    def forward(self, input_ids, attention_mask, labels, cur_signal, ref_signal):
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        
        ref_feature = self.encoder(ref_signal)
        ref_embed = self.projector(ref_feature).unsqueeze(1)
        ref_mask = (input_ids == self.vib_token_id).cumsum(dim=1) == 1
        input_embeds = torch.where(ref_mask.unsqueeze(-1), ref_embed, input_embeds)
        
        cur_feature = self.encoder(cur_signal)
        cur_embed = self.projector(cur_feature).unsqueeze(1)
        cur_mask = (input_ids == self.vib_token_id).cumsum(dim=1) == 2
        input_embeds = torch.where(cur_mask.unsqueeze(-1), cur_embed, input_embeds)
        
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

            Reference vibration state: <VIB_EMB>
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
            "cur_signal": signal_tensor,  # vibration input for embedding injection
            "ref_signal": ref_tensor
        }

class TrainerModelWrapper(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model

    def forward(self, input_ids, attention_mask, labels, vibration_signal):
        return self.model(input_ids, attention_mask, vibration_signal, labels=labels)
    
def data_collator(features):
    input_ids = torch.stack([f['input_ids'] for f in features])
    attention_mask = torch.stack([f['attention_mask'] for f in features])
    labels = torch.stack([f['labels'] for f in features])
    cur_signal = torch.stack([f['cur_signal'] for f in features])
    ref_signal = torch.stack([f['ref_signal'] for f in features])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "cur_signal": cur_signal,  # vibration input for embedding injection
        "ref_signal": ref_signal
    }

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 진동에 학습된 모델 불러오기
    unet_vae = UnetVAE(
        in_length               =256, 
        in_channels             =2, 
        num_residual_layers     =8, 
        num_residual_hiddens    =2, 
        num_embeddings          =128, 
        embedding_dim           =32
    )
    vib_encoder = unet_vae.encoder.to(device)
    vibration_dataset = VibrationDataset(
        data_root='/workspace/dataset',
        statistic_info=True
    )
    
    # LLM 모델 불러오기
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': ['<VIB_EMB>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if bfloat16 불가
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # 또는 torch.float16
        quantization_config=bnb_config,
        device_map=None  # ✅ device_map 아예 사용하지 말 것!
    )
    llm.resize_token_embeddings(len(tokenizer))
    print(f'llm toekn size : {llm.config.hidden_size}')
    
    
    for param in vib_encoder.parameters():
        param.requires_grad = False
    projector = Projector(
        in_dim= int(128 * 32),
        output_dim=llm.config.hidden_size).to(device)
    vib_token_id = tokenizer.convert_tokens_to_ids('<VIB_EMB>')
    
    full_model = FullModel(vib_encoder, projector, llm, vib_token_id).to(device)
    
    dataset = VibrationSFTDatasetEnglish(
        tokenizer = tokenizer,
        vibration_dataset= vibration_dataset,
        max_length=2048
    )
    
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_steps=10,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        save_steps=10,
        save_total_limit=2,
        logging_steps=1,
        report_to="none"
    )
    
    trainer = Trainer(
        model=full_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()