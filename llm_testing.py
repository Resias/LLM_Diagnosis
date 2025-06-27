import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from data.dataset import VibrationDataset
from models.unet import UnetVAE
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics.classification import MulticlassAccuracy

import csv
import os

# Projector Module
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

# Dataset
class VibrationSFTDatasetEnglish(Dataset):
    def __init__(self, tokenizer, vibration_dataset, max_length=1024):
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

        system_prompt = (
            "Act as an expert in vibration-based diagnosis of rotating machinery. "
            "Analyze the statistical features of both the reference (normal) and current vibration signals I provide, "
            "and accurately diagnose the current condition of the equipment. "
            "Possible diagnostic classes are: looseness, normal, unbalance, misalignment, bearing."
        )

        user_prompt = (
            f"Machine running at {signal_info['rpm']} RPM under load condition: "
            f"{signal_info['load_condition'] if signal_info['load_condition'] != 'unknown' else 'unknown'}.\n"
            f"Reference state (time domain x): {ref_time_feature_dict[0]}\n"
            f"Reference state (time domain y): {ref_time_feature_dict[1]}\n"
            f"Reference state (freq domain x): {ref_freq_feature_dict[0]}\n"
            f"Reference state (freq domain y): {ref_freq_feature_dict[1]}\n"
            f"Current state (time domain x): {time_feature_dict[0]}\n"
            f"Current state (time domain y): {time_feature_dict[1]}\n"
            f"Current state (freq domain x): {freq_feature_dict[0]}\n"
            f"Current state (freq domain y): {freq_feature_dict[1]}\n"
            "Reference state embedding: <VIB_EMB>. Current state embedding: <VIB_EMB>. "
            "Diagnose condition: looseness, normal, unbalance, misalignment, bearing."
        )

        # Generate a detailed CoT reasoning string emphasizing <VIB_EMB>
        reasoning_part = (
            "Reasoning:\n"
            "Step 1: Analyze the changes in time-domain features (e.g., kurtosis, skewness, peak amplitude, etc).\n"
            "Step 2: Analyze the changes in frequency-domain features (e.g., spectral entropy, spectral kurtosis, dominant frequency power, etc).\n"
            "Step 3: Compare embeddings of reference and current signals to detect overall change patterns.\n"
            "Step 4: Integrate all statistical and embedding-based evidence to formulate hypothesis.\n"
        )
        final_answer = f"Answer: The diagnosis result is {signal_info['class_name']}."
        assistant_response = f"{reasoning_part}\n{final_answer}"
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: {assistant_response}"

        encoded = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        prompt_len = len(self.tokenizer(prompt_only, return_tensors='pt')['input_ids'][0])
        labels[:prompt_len] = -100

        class_map = {"looseness": 0, "normal": 1, "unbalance": 2, "misalignment": 3, "bearing": 4}
        class_label = class_map[signal_info['class_name']]

        return {
            "input_ids": input_ids,
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "labels": labels,
            "cur_signal": signal_tensor,
            "ref_signal": ref_tensor,
            "class_label": torch.tensor(class_label, dtype=torch.long)
        }

# Lightning Model
class LightningFullModel(pl.LightningModule):
    def __init__(self, encoder, projector, llm, vib_token_id, label_classes, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.vib_token_id = vib_token_id
        self.label_classes = label_classes
        self.tokenizer = tokenizer
        self.correct = 0
        self.total = 0

    def extract_class_from_output(self, output_text):
        output_text = output_text.lower()
        import re
        match = re.search(r"diagnosis result is[:\- ]*\s*(\w+)", output_text)
        if match:
            candidate = match.group(1).strip()
            for cls in self.label_classes:
                if cls in candidate:
                    return cls
        for cls in self.label_classes:
            if cls in output_text:
                return cls
        return "unknown"

    def replace_vib_tokens_with_embedding(self, input_ids, ref_embed, cur_embed):
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        positions = (input_ids == self.vib_token_id).nonzero(as_tuple=False)
        for b in range(input_ids.size(0)):
            batch_positions = positions[positions[:, 0] == b][:, 1]
            if batch_positions.numel() >= 1:
                input_embeds[b, batch_positions[0], :] = ref_embed[b]
            if batch_positions.numel() >= 2:
                input_embeds[b, batch_positions[1], :] = cur_embed[b]
        return input_embeds

    def forward(self, input_ids, attention_mask, labels, cur_signal, ref_signal):
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        ref_feature = self.encoder(ref_signal)
        ref_embed = self.projector(ref_feature)
        cur_feature = self.encoder(cur_signal)
        cur_embed = self.projector(cur_feature)

        positions = (input_ids == self.vib_token_id).nonzero(as_tuple=False)
        for batch_idx in range(input_ids.size(0)):
            batch_positions = positions[positions[:, 0] == batch_idx][:, 1]
            if batch_positions.numel() >= 1:
                input_embeds[batch_idx, batch_positions[0], :] = ref_embed[batch_idx]
            if batch_positions.numel() >= 2:
                input_embeds[batch_idx, batch_positions[1], :] = cur_embed[batch_idx]

        outputs = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            cur_signal=batch["cur_signal"],
            ref_signal=batch["ref_signal"]
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        if self.global_rank == 0:
            self.log_file = "validation_predictions.csv"
            write_header = not os.path.exists(self.log_file)
            self.csv_file = open(self.log_file, mode="a", newline='', encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            if write_header:
                self.csv_writer.writerow(["epoch", "gt_label", "generated_output", "predicted_class"])
        # Limit number of logged samples per epoch
        self.logged_samples = 0
        self.max_log_samples = 10

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        ref_feature = self.encoder(batch["ref_signal"])
        ref_embed = self.projector(ref_feature)
        cur_feature = self.encoder(batch["cur_signal"])
        cur_embed = self.projector(cur_feature)

        # Replace vib tokens in input_ids directly for safe context tracking
        for b in range(input_ids.size(0)):
            vib_positions = (input_ids[b] == self.vib_token_id).nonzero(as_tuple=False).flatten()
            if vib_positions.numel() >= 1:
                token_embed = self.llm.get_input_embeddings().weight.data
                token_embed[self.vib_token_id] = ref_embed[b]
            if vib_positions.numel() >= 2:
                token_embed = self.llm.get_input_embeddings().weight.data
                token_embed[self.vib_token_id] = cur_embed[b]

        generated = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,  # increased from 100
            # temperature=0.7,
            # top_p=0.9,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for i, output_text in enumerate(decoded):
            gt_label = batch["class_label"][i].item()
            predicted_class = self.extract_class_from_output(output_text)

            # If output is too short, apply CoT prompting
            if len(output_text.split()) < 20:
                cot_prompt = output_text + "\nExplain step by step how you arrived at this diagnosis before answering."
                cot_encoded = self.tokenizer(
                    cot_prompt, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=self.tokenizer.model_max_length
                ).to(self.device)

                cot_generated = self.llm.generate(
                    input_ids=cot_encoded['input_ids'],
                    attention_mask=cot_encoded['attention_mask'],
                    max_new_tokens=400,  # increased from 150
                    # temperature=0.7,
                    # top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                cot_decoded = self.tokenizer.batch_decode(cot_generated, skip_special_tokens=True)[0]
                predicted_class = self.extract_class_from_output(cot_decoded)
                output_text = cot_decoded

            if self.global_rank == 0 and self.logged_samples < self.max_log_samples:
                assistant_part = output_text.split("Assistant:")[-1].strip()
                self.csv_writer.writerow([self.current_epoch, self.label_classes[gt_label], assistant_part, predicted_class])
                self.logged_samples += 1
            if predicted_class == self.label_classes[gt_label]:
                self.correct += 1
            self.total += 1

    def on_validation_epoch_end(self):
        acc = self.correct / self.total if self.total > 0 else 0.0
        self.log("val_classification_acc", acc, prog_bar=True)
        if self.global_rank == 0:
            self.csv_file.flush()
            self.csv_file.close()
        self.correct = 0
        self.total = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

def collate_fn(features):
    return {
        "input_ids": torch.stack([f['input_ids'] for f in features]),
        "attention_mask": torch.stack([f['attention_mask'] for f in features]),
        "labels": torch.stack([f['labels'] for f in features]),
        "cur_signal": torch.stack([f['cur_signal'] for f in features]),
        "ref_signal": torch.stack([f['ref_signal'] for f in features]),
        "class_label": torch.stack([f['class_label'] for f in features]),
    }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_vae = UnetVAE(
        in_length=256, in_channels=2, num_residual_layers=8,
        num_residual_hiddens=2, num_embeddings=128, embedding_dim=32
    )
    unet_vae.load_state_dict(torch.load('./model_saved/ed_32_ne_128_nrl_8_nrh_2_model.pth'))
    vib_encoder = unet_vae.encoder.to(device)
    for param in vib_encoder.parameters():
        param.requires_grad = False

    vibration_dataset = VibrationDataset(data_root='/workspace/dataset', statistic_info=True)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<VIB_EMB>']})
    tokenizer.pad_token = tokenizer.eos_token

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

    projector = Projector(in_dim=128*32, output_dim=llm.config.hidden_size).to(device)
    vib_token_id = tokenizer.convert_tokens_to_ids('<VIB_EMB>')

    label_classes = ["looseness", "normal", "unbalance", "misalignment", "bearing"]
    full_model = LightningFullModel(vib_encoder, projector, llm, vib_token_id, label_classes, tokenizer)

    full_dataset = VibrationSFTDatasetEnglish(tokenizer, vibration_dataset)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=15)

    trainer = pl.Trainer(
        accelerator="gpu", devices=torch.cuda.device_count(),
        precision=16, max_epochs=200
    )

    trainer.fit(full_model, train_loader, val_loader)