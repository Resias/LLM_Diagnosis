from trl import GRPOTrainer, GRPOConfig
from typing import Any, Union
import torch
from accelerate.utils import gather_object
from trl.data_utils import maybe_apply_chat_template
import re
import copy
import json
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
from trl.models import unwrap_model_for_generation
from html import escape
import wandb


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
    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        vib_tokenizer=None,   # 진동 토크나이저(Custom)
    ):
        # 부모 초기화 그대로
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        # 내가 추가한 것만 저장
        self.vib_tokenizer = vib_tokenizer
        
    """
    GRPO에 custom multi-modal 기능이 없어 구현
    """

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        prompts = [x["prompt"] for x in inputs]
        
        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)
        
        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left"
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # 스페셜 토큰의 위치를 찾아 저장한다. (Custom)
        normal_id = self.processing_class.convert_tokens_to_ids("<NORMAL_VIB_EMB>")
        current_id = self.processing_class.convert_tokens_to_ids("<CURRENT_VIB_EMB>")
        

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

            # 진동 토큰 주입 파트(Custom)
            for i in range(len(inputs)):
                # Find positions of placeholder tokens in the tokenized prompt
                # Use as_tuple=True to get a 1D index tensor and handle 0/1/Many occurrences robustly.
                n_pos = (prompt_ids[i] == normal_id).nonzero(as_tuple=True)[0]
                c_pos = (prompt_ids[i] == current_id).nonzero(as_tuple=True)[0]

                # If placeholders are missing (e.g., due to truncation), skip replacement to avoid crashes.
                if n_pos.numel() == 0 or c_pos.numel() == 0:
                    assert f"[warn] Placeholder token(s) not found in sample {i}. "
                    exit()

                # Assign the same learned vibration embedding to all matching positions
                # Ensure dtype/device match the model input embeddings
                normal_x = inputs[i]['normal_x'].to(device=self.vib_tokenizer.device, dtype=self.vib_tokenizer.dtype)
                current_x = inputs[i]['current_x'].to(device=self.vib_tokenizer.device, dtype=self.vib_tokenizer.dtype)
                
                n_emb = self.vib_tokenizer(normal_x)
                n_emb = n_emb.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                c_emb = self.vib_tokenizer(current_x)
                c_emb = c_emb.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                
                # Expand to match the number of positions if there are multiple matches
                n_emb = n_emb.unsqueeze(0) if n_emb.dim() == 1 else n_emb
                c_emb = c_emb.unsqueeze(0) if c_emb.dim() == 1 else c_emb
                input_embeddings[i, n_pos] = n_emb.expand(n_pos.shape[0], -1)
                input_embeddings[i, c_pos] = c_emb.expand(c_pos.shape[0], -1)

            prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
            prompt_completion_ids = unwrapped_model.generate(
                inputs_embeds=input_embeddings, attention_mask=prompt_mask, generation_config=self.generation_config
            )

        # Compute prompt length and extract completion ids 
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
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
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None
                
            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # 여기에 생성부분만 남기도록 후처리 필요 (Custom)
        completions = completions_text
        print(f'prompt : {prompts}')
        print(f'response : {completions}')
        
        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

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
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output
        

def reward_accuracy(completions, answers, **kwargs):
    """
    정답 보상: <answer>{...}</answer> 내부 JSON에서 final_label을 파싱하여 정답과 일치하면 1.0, 아니면 0.0
    answers는 데이터셋이 제공하는 정답(문자열)로 가정.
    """
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        try:
            text = completion["content"] if isinstance(completion, dict) else completion
            m = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
            answer_body = m.group(1).strip() if m else ""
            # JSON 파싱 시도
            parsed = None
            try:
                parsed = json.loads(answer_body)
            except Exception:
                # 마지막 쉼표 등 사소한 오류가 있을 수 있어, 최후 수단: 정규식으로 final_label만 추출
                mm = re.search(r'\"final_label\"\s*:\s*\"([^\"]+)\"', answer_body)
                if mm:
                    parsed = {"final_label": mm.group(1)}
            final_label = (parsed or {}).get("final_label", "")
            is_correct = str(final_label).strip().lower() == str(correct_answer).strip().lower()
            rewards.append(1.0 if is_correct else 0.0)
        except Exception as e:
            print(f"[Reward Parse Error] {e}")
            rewards.append(0.0)
    return rewards

def reward_format(completions, **kwargs):
    """
    형식 보상: 아래 조건을 모두 만족하면 1.0, 아니면 0.0
    - 정확히 하나의 <reasoning>...</reasoning> 블록 존재
    - 정확히 하나의 <answer>{JSON}</answer> 블록 존재 (코드펜스 금지)
    - answer JSON에 필수 키 존재: vib_only_label, vib_reason, knowledge_only_label, knowledge_reason, final_label, fusion_reason
    - 중복 답변/여분 텍스트 없음 (블록 외 불필요한 텍스트가 실질적으로 없어야 함)
    """
    required_keys = {
        "vib_only_label",
        "vib_reason",
        "knowledge_only_label",
        "knowledge_reason",
        "final_label",
        "fusion_reason",
    }
    rewards = []
    for completion in completions:
        try:
            text = completion["content"] if isinstance(completion, dict) else completion
            s = text.strip()
            # 코드펜스가 있으면 실패
            if "```" in s:
                rewards.append(0.0)
                continue
            # reasoning / answer 매칭
            reasoning_matches = re.findall(r"<reasoning>(.*?)</reasoning>", s, re.DOTALL)
            answer_matches = re.findall(r"<answer>(\{.*?\})</answer>", s, re.DOTALL)
            if len(reasoning_matches) != 1 or len(answer_matches) != 1:
                rewards.append(0.0)
                continue
            # answer JSON 파싱 및 키 검증
            ok = False
            try:
                answer_obj = json.loads(answer_matches[0])
                ok = required_keys.issubset(set(answer_obj.keys()))
            except Exception:
                ok = False
            if not ok:
                rewards.append(0.0)
                continue
            # 중복/여분 텍스트 최소화: 태그 제거 후 공백만 남아야 하는지 검사(관대하게 공백/개행 허용)
            cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", s, flags=re.DOTALL)
            cleaned = re.sub(r"<answer>.*?</answer>", "", cleaned, flags=re.DOTALL)
            if cleaned.strip() != "":
                # 여분 텍스트가 있으면 0.0 (중복/군더더기 방지)
                rewards.append(0.0)
                continue
            rewards.append(1.0)
        except Exception as e:
            print(f"[Format Parse Error] {e}")
            rewards.append(0.0)
    return rewards
