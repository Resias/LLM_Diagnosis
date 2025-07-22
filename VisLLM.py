import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Union
from accelerate.utils import gather_object
from trl.data_utils import maybe_apply_chat_template
import re
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
from data.order_dataset import OrderFreqDataset
from models.segment_transformer_recon import SegmentReconModel

from trl.models import unwrap_model_for_generation
from trl import GRPOTrainer, GRPOConfig
from peft import get_peft_model, LoraConfig, TaskType
from utils.reward import reward_format, reward_accuracy

import wandb
from html import escape

class VibEmbedding(nn.Module):
    def __init__(self, vib_encoder, token_embed_dim, freeze_encoder=True):
        super().__init__()
        self.vib_encoder = vib_encoder
        device = next(self.vib_encoder.parameters()).device

        if freeze_encoder:
            for param in self.vib_encoder.parameters():
                param.requires_grad = False

        in_dim = int(self.vib_encoder.embed_dim * 10)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_dim,
                out_features=token_embed_dim
            )
        ).to(device)

    def forward(self, x_current, x_normal):
        device = next(self.model.parameters()).device
        x_current = x_current.to(device)
        x_normal = x_normal.to(device)
        sample_attn, normal_attn, attension_dict = self.vib_encoder(x_current, x_normal, get_z=True)
        current_z = self.model(sample_attn)
        normal_z = self.model(normal_attn)
        return current_z, normal_z

# Dataset
class VibrationSFTDatasetEnglish(Dataset):
    def __init__(self, vibration_dataset, vib_encoder, embedding_dim):
        self.vibration_dataset = vibration_dataset
        self.vib_tokenizer = VibEmbedding(
            vib_encoder = vib_encoder,
            token_embed_dim=embedding_dim
        )

    def __len__(self):
        return len(self.vibration_dataset)

    def __getitem__(self, idx):
        (
            sample_tensor, normal_tensor, row
        ) = self.vibration_dataset.__getitem__(idx, data_info=True)
        
        sample_tensor = sample_tensor.unsqueeze(0)
        normal_tensor = normal_tensor.unsqueeze(0)
        currnet_token, normal_token = self.vib_tokenizer(sample_tensor, normal_tensor)
        currnet_token = currnet_token.squeeze(0).cpu()
        normal_token = normal_token.squeeze(0).cpu()
        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer." 
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively," 
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        )

        # Update user_prompt with retrieved context
        user_prompt = (
            "You are an expert in vibration-based diagnosis of rotating machinery. "
            "Given the both the normal and current vibration signals, "
            "provide a concise diagnosis. Possible conditions: looseness, normal, unbalance, misalignment, bearing.\n"
            f"Normal state : <NORMAL_VIB_EMB>\n"
            f"Current state : <CURRENT_VIB_EMB>\n"
        )
        
        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        assistant_response = f"The diagnosis result is {row['class_name']}."
        
        return {
            "prompt": prompt_only,
            "answers": assistant_response,
            "normal_token" : normal_token,
            "currnet_token" : currnet_token
        }

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class CustomGRPOTRainer(GRPOTrainer):
    """
    GRPO에 custom multi-modal 기능이 없어 구현
    """
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # print(inputs[0]['prompt'])
        # for data_sample in inputs:
        #     print(data_sample['prompt'])
        
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left"
        )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        normal_id = self.processing_class.convert_tokens_to_ids("<NORMAL_VIB_EMB>")
        current_id = self.processing_class.convert_tokens_to_ids("<CURRENT_VIB_EMB>")
        
        
        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [
                re.sub(rf"^({re.escape(self.processing_class.pad_token)})+", "", text) for text in prompts_text
            ]

        # Regular generation path
        with (
            # profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            # Prepare input embeddings and replace special tokens with vibration embeddings
            prompt_ids = prompt_ids.to(self.model.device)
            input_embeddings = self.model.get_input_embeddings()(prompt_ids.clone())
            prompt_mask = prompt_mask.to(self.model.device)

            for i in range(len(inputs)):
                n_pos = (prompt_ids[i] == normal_id).nonzero(as_tuple=False)
                c_pos = (prompt_ids[i] == current_id).nonzero(as_tuple=False)
 
                input_embeddings[i, n_pos.item()] = inputs[i]['normal_token'].to(input_embeddings.device)
                input_embeddings[i, c_pos.item()] = inputs[i]['currnet_token'].to(input_embeddings.device)

            
            prompt_completion_ids = unwrapped_model.generate(
                inputs_embeds=input_embeddings, attention_mask=prompt_mask, generation_config=self.generation_config
            )

            # Log one example to wandb as HTML every 100 steps
            if self.state.global_step % 100 == 0:
                sample_prompt = self.processing_class.decode(prompt_ids[0], skip_special_tokens=False)
                sample_output = self.processing_class.decode(prompt_completion_ids[0], skip_special_tokens=False)

                html_str = f"""
                <div style='font-family:monospace'>
                    <b>Prompt:</b><br>
                    <pre>{escape(sample_prompt)}</pre><br>
                    <b>Output:</b><br>
                    <pre>{escape(sample_output)}</pre>
                </div>
                """

                wandb.log({"generation_html": wandb.Html(html_str)}, step=self.state.global_step)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )["logps"]
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )["logps"]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )["logps"]
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

if __name__ == '__main__':
    
    device = 'cuda'
    
    #승하가 학습시킨 모델 가중치 로드할 수 있도록 변경
    vib_encoder = SegmentReconModel().to(device)
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir='./cache_models/r1')
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 cache_dir='./cache_models/r1').to(device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)
    
    
    special_tokens = {
    'additional_special_tokens': ["<NORMAL_VIB_EMB>", "<CURRENT_VIB_EMB>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    llm.resize_token_embeddings(len(tokenizer))
    
    train_orderdatsaet = OrderFreqDataset(
                data_root = '/workspace/dataset', 
                dataset_list = ['dxai', 'mfd', 'vbl'],
    )
    trainset = VibrationSFTDatasetEnglish(
        vibration_dataset=train_orderdatsaet,
        vib_encoder = vib_encoder,
        embedding_dim = llm.config.hidden_size
    )
    val_orderdatsaet = OrderFreqDataset(
                data_root = '/workspace/dataset', 
                dataset_list = ['vat'],
    )
    valset = VibrationSFTDatasetEnglish(
        vibration_dataset=val_orderdatsaet,
        vib_encoder = vib_encoder,
        embedding_dim = llm.config.hidden_size
    )
    

    # training_args = GRPOConfig(
    #     output_dir="output",
    #     logging_strategy="steps",
    #     logging_steps=10,
    #     report_to="wandb",
    # )
    # trainer = CustomGRPOTRainer(
    #     model=llm,
    #     processing_class=tokenizer,
    #     args=training_args,
    #     train_dataset=trainset,
    #     eval_dataset=valset,
    #     reward_funcs=[reward_format, reward_accuracy],
    # )

    # for name, param in trainset.vib_tokenizer.vib_encoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"[ENCODER] Trainable: {name}")
    # for name, param in trainset.vib_tokenizer.model.named_parameters():
    #     if param.requires_grad:
    #         print(f"[VIB_EMBEDDING] Trainable: {name}")

    # vib_embedder = trainset.vib_tokenizer
    # model = trainer.model
    # learning_rate = training_args.learning_rate
    # params = list(filter(lambda p: p.requires_grad, vib_embedder.parameters())) + \
    #          list(filter(lambda p: p.requires_grad, model.parameters()))
    # trainer.optimizer = torch.optim.AdamW(params, lr=learning_rate)

    # trainer.train()