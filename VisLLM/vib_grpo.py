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
from trl.trainer.utils import selective_log_softmax

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
        model.generation_config.eos_token_id = processing_class.convert_tokens_to_ids("</answer>")
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
        self.ans_ids = self.processing_class.convert_tokens_to_ids("</answer>")
        self.eos_token_id = self.processing_class.convert_tokens_to_ids("</answer>")
    
    # 🚨 [수정 포인트 1] 모델 순전파(forward pass) 로직 중앙화
    # 커스텀 임베딩 주입과 시퀀스 길이 문제를 이 메서드에서 모두 해결합니다.

    def _get_per_token_logps_and_entropies(
        self,
        model,
        completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        **kwargs,
    ):
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        normal_x = kwargs.pop("normal_x", None)
        current_x = kwargs.pop("current_x", None)

        # 입력 텐서들을 logits_to_keep에 맞춰 미리 슬라이싱
        input_ids = completion_ids[:, -logits_to_keep:]
        input_mask = attention_mask[:, -logits_to_keep:]
        
        input_embeddings = unwrapped_model.get_input_embeddings()(input_ids)
            
        if normal_x is not None and current_x is not None:
            normal_id = self.processing_class.convert_tokens_to_ids("<NORMAL_VIB_EMB>")
            current_id = self.processing_class.convert_tokens_to_ids("<CURRENT_VIB_EMB>")
            
            # 🚨 [최종 수정] 루프의 기준과 내부 로직을 슬라이싱된 'input_ids'로 통일합니다.
            for i in range(len(input_ids)):
                n_pos = (input_ids[i] == normal_id).nonzero(as_tuple=True)[0]
                c_pos = (input_ids[i] == current_id).nonzero(as_tuple=True)[0]

                if n_pos.numel() > 0 and c_pos.numel() > 0:
                    n_emb = self.vib_tokenizer(normal_x[i]).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                    c_emb = self.vib_tokenizer(current_x[i]).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                    # input_embeddings의 해당 위치에 직접 주입
                    input_embeddings[i, n_pos] = n_emb.expand(n_pos.shape[0], -1)
                    input_embeddings[i, c_pos] = c_emb.expand(c_pos.shape[0], -1)

        outputs = model(inputs_embeds=input_embeddings, attention_mask=input_mask, use_cache=False, **kwargs)
        logits = outputs.logits

        logps = selective_log_softmax(logits, input_ids)
        
        all_logps = logits.log_softmax(dim=-1)
        all_probs = torch.exp(all_logps)
        entropies = -torch.sum(all_probs * all_logps, dim=-1)

        return logps, entropies
    

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
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
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
            # --- Generation을 위한 임베딩 주입 ---
            prompt_ids = prompt_ids.to(self.model.device)
            input_embeddings = self.model.get_input_embeddings()(prompt_ids)
            prompt_mask = prompt_mask.to(self.model.device)

            normal_id = self.processing_class.convert_tokens_to_ids("<NORMAL_VIB_EMB>")
            current_id = self.processing_class.convert_tokens_to_ids("<CURRENT_VIB_EMB>")

            for i in range(len(inputs)):
                n_pos = (prompt_ids[i] == normal_id).nonzero(as_tuple=True)[0]
                c_pos = (prompt_ids[i] == current_id).nonzero(as_tuple=True)[0]
                if n_pos.numel() == 0 or c_pos.numel() == 0: continue
                
                normal_x = inputs[i]['normal_x'].to(device=self.vib_tokenizer.device, dtype=self.vib_tokenizer.dtype)
                current_x = inputs[i]['current_x'].to(device=self.vib_tokenizer.device, dtype=self.vib_tokenizer.dtype)
                n_emb = self.vib_tokenizer(normal_x).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                c_emb = self.vib_tokenizer(current_x).to(device=input_embeddings.device, dtype=input_embeddings.dtype)
                input_embeddings[i, n_pos] = n_emb.expand(n_pos.shape[0], -1)
                input_embeddings[i, c_pos] = c_emb.expand(c_pos.shape[0], -1)
            
            prompt_completion_ids = unwrapped_model.generate(
                inputs_embeds=input_embeddings, attention_mask=prompt_mask, generation_config=self.generation_config
            )
            
        # --- 생성 후 처리 및 점수 계산 ---
        completion_ids = prompt_completion_ids
        
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = completion_mask
        

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [[id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)]
        completion_lengths = completion_mask.sum(1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        
        # 커스텀 데이터를 텐서로 변환
        normal_x_tensor = torch.stack([inp['normal_x'] for inp in inputs])
        current_x_tensor = torch.stack([inp['current_x'] for inp in inputs])


        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                # 🚨 오버라이드한 중앙화된 메서드를 호출합니다. (커스텀 데이터 전달)
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep,
                    normal_x=normal_x_tensor, current_x=current_x_tensor
                )
            else:
                old_per_token_logps = None
            
            if self.beta != 0.0:
                if self.ref_model is not None:
                    # 🚨 ref_model은 커스텀 임베딩을 모르므로, 부모의 원본 메서드를 호출합니다.
                    ref_per_token_logps, _ = super()._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else: # PEFT 어댑터를 사용하는 경우
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = super()._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # 프롬프트 부분 제거하여 순수 completion만 남기기
        pure_completions_text = []
        for i, comp_text in enumerate(completions_text):
            prompt_text = prompts_text[i]
            # prompt가 completion의 시작 부분에 있다면 제거
            if comp_text.startswith(prompt_text):
                pure_completions_text.append(comp_text[len(prompt_text):])
            else:
                pure_completions_text.append(comp_text)
        
        # 여기에 생성부분만 남기도록 후처리 필요 (Custom)
        completions = pure_completions_text
        
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

        if self.state.global_step > 0 and self.state.global_step % 10 == 0 and self.accelerator.is_main_process:
            # 1. 가장 핵심적인 정보만 담을 테이블을 생성합니다.
            table = wandb.Table(columns=["Global Step", "Completion", "Total Reward"])
            
            # 2. 로깅할 샘플 수를 정합니다 (예: 2-4개).
            num_samples_to_log = min(len(completions), 4)
            
            for i in range(num_samples_to_log):
                # 생성 결과와 보상 점수를 가져옵니다.
                completion = completions[i]
                total_reward = sum(rewards_per_func[i, j].item() for j in range(len(self.reward_func_names)))
                
                # 테이블에 데이터 한 줄을 추가합니다.
                table.add_data(
                    self.state.global_step,
                    completion,
                    total_reward
                )
            
            # 3. 테이블을 WandB에 로깅합니다.
            wandb.log({"generation_examples": table})

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
         # 다음 단계(compute_loss)에서 _get_per_token_logps_and_entropies를 호출할 때 필요
        output["normal_x"] = normal_x_tensor
        output["current_x"] = current_x_tensor
        
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
    

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        old_per_token_logps = inputs["old_per_token_logps"]
        ref_per_token_logps = inputs.get("ref_per_token_logps")
        advantages = inputs["advantages"]
        completion_ids = inputs["completion_ids"]
        attention_mask = inputs["completion_mask"]

        response_length = old_per_token_logps.shape[-1]
        logits_to_keep = response_length + 1
        
        known_args = {
            "completion_ids", "attention_mask", "advantages", "old_per_token_logps",
            "ref_per_token_logps", "prompt_ids", "prompt_mask"
        }
        extra_kwargs = {k: v for k, v in inputs.items() if k not in known_args}
        
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model,
            completion_ids,
            attention_mask,
            logits_to_keep,
            **extra_kwargs,
        )

        log_ratio = per_token_logps - old_per_token_logps
        
        if self.beta != 0.0 and ref_per_token_logps is not None:
            kl_penalty = self.beta * (per_token_logps - ref_per_token_logps)
            log_ratio = log_ratio - kl_penalty

        completion_log_ratio = log_ratio

        expanded_advantages = advantages.unsqueeze(1).expand(-1, response_length)
        completion_advantages = expanded_advantages
        
        # 🚨 [최종 수정] 마스크를 슬라이싱할 때 올바른 시작 인덱스를 사용합니다.
        prompt_length = inputs["prompt_ids"].size(1)
        start = prompt_length  # 👈 -1을 제거하여 올바른 시작점으로 수정
        end = start + response_length
        completion_mask = attention_mask[:, start:end]
        
        # 텐서 크기가 일치하는지 최종 확인 (디버깅용)
        if completion_log_ratio.shape != completion_mask.shape:
            # 길이를 맞추기 위한 예외 처리 (예: completion_mask가 더 긴 경우)
            target_len = min(completion_log_ratio.shape[1], completion_mask.shape[1])
            completion_log_ratio = completion_log_ratio[:, :target_len]
            completion_advantages = completion_advantages[:, :target_len]
            completion_mask = completion_mask[:, :target_len]

        # 패딩된 토큰의 영향을 받지 않도록 마스크를 곱해줍니다.
        loss = -0.5 * (completion_log_ratio * completion_advantages * completion_mask).sum() / completion_mask.sum()
        
        return loss if not return_outputs else (loss, {"loss": loss})

def reward_accuracy(completions, answers, **kwargs):
    """
    세분화된 정답 보상:
    - vib_only_label 정답 시: +0.2점
    - knowledge_only_label 정답 시: +0.2점
    - final_label 정답 시: +0.6점
    최대 1.0점, 모두 틀리면 0.0점
    """
    rewards = []
    for completion, correct_answers_dict in zip(completions, answers):
        try:
            current_reward = 0.0
            text = completion["content"] if isinstance(completion, dict) else completion

            # <answer> 블록에서 JSON 파싱
            m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text or "", re.DOTALL)
            if not m:
                rewards.append(0.0)
                continue
            
            answer_body = m.group(1).strip()
            parsed = json.loads(answer_body)

            # 1. vib_only_label 확인
            vib_label = parsed.get("vib_only_label", "").strip().lower()
            correct_vib_label = correct_answers_dict.get("vib_only_label", "").strip().lower()
            if vib_label and vib_label == correct_vib_label:
                current_reward += 0.2

            # 2. knowledge_only_label 확인
            knowledge_label = parsed.get("knowledge_only_label", "").strip().lower()
            correct_knowledge_label = correct_answers_dict.get("knowledge_only_label", "").strip().lower()
            if knowledge_label and knowledge_label == correct_knowledge_label:
                current_reward += 0.2

            # 3. final_label 확인
            final_label = parsed.get("final_label", "").strip().lower()
            correct_final_label = correct_answers_dict.get("final_label", "").strip().lower()
            if final_label and final_label == correct_final_label:
                current_reward += 0.6
            
            rewards.append(current_reward)
            
        except Exception as e:
            # print(f"[Reward Parse Error] {e}")
            rewards.append(0.0)
            
    return rewards


def reward_format(completions, **kwargs):
    required_keys = {
        "vib_only_label", "vib_reason", "knowledge_only_label",
        "knowledge_reason", "final_label", "fusion_reason"
    }
    rewards = []
    for i, completion in enumerate(completions):
        try:
            # print(f"\n--- [DEBUG] CHECKING COMPLETION #{i+1} ---")
            text = completion["content"] if isinstance(completion, dict) else completion
            s = text.strip()

            # 체크 1: 코드펜스
            if "```" in s:
                # print("FAIL: Found ``` backticks.")
                rewards.append(0.0)
                continue
            # print("PASS: No backticks.")

            # 체크 2: 블록 개수
            reasoning_matches = re.findall(r"<reasoning>(.*?)</reasoning>", s, re.DOTALL)
            answer_matches = re.findall(r"<answer>\s*(\{.*?\})\s*</answer>", s, re.DOTALL)
            if len(reasoning_matches) != 1 or len(answer_matches) != 1:
                # print(f"FAIL: Block count mismatch. Reasoning: {len(reasoning_matches)}, Answer: {len(answer_matches)}")
                rewards.append(0.0)
                continue
            # print("PASS: Correct block count (Reasoning: 1, Answer: 1).")

            # 체크 3: JSON 파싱 및 키 검증
            ok = False
            try:
                answer_obj = json.loads(answer_matches[0])
                # print("PASS: JSON parsed successfully.")
                
                missing_keys = required_keys - set(answer_obj.keys())
                if not missing_keys:
                    ok = True
                # else:
                    # print(f"FAIL: Required keys are missing: {missing_keys}")

            except Exception as e:
                # print(f"FAIL: JSON parse error: {e}")
                ok = False

            if not ok:
                rewards.append(0.0)
                continue
            # print("PASS: All required JSON keys are present.")
            
            # 체크 4: 불필요한 외부 텍스트
            cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", s, flags=re.DOTALL)
            cleaned = re.sub(r"<answer>.*?</answer>", "", cleaned, flags=re.DOTALL)
            if cleaned.strip() != "":
                # print(f"FAIL: Extra text found outside blocks: '{cleaned.strip()}'")
                rewards.append(0.0)
                continue
            # print("PASS: No extra text found.")

            # 모든 체크 통과
            # print("SUCCESS: All format checks passed!")
            rewards.append(1.0)

        except Exception as e:
            # print(f"[CRITICAL ERROR] in reward_format: {e}")
            rewards.append(0.0)
            
    # print(f"--- [DEBUG] FINAL REWARDS: {rewards} ---\n")
    return rewards

