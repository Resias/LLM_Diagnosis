import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional
import re
# ===== RAG Helpers: extract physical info from knowledge strings, build queries, summarize retrieved docs =====
def _parse_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default

def _extract_physical_info(knowledge: str) -> dict:
    """
    Extract simple physical indicators from the structured knowledge string produced by _build_knowledge_string.
    Returns keys: rms_x, rms_y, kurt_x, kurt_y, cf_x, cf_y, near_1x, near_2x, near_3x, rpm, f_rot, top_peaks_hz(list), orders(list)
    """
    info = {
        'rms_x': None, 'rms_y': None,
        'kurt_x': None, 'kurt_y': None,
        'cf_x': None, 'cf_y': None,
        'near_1x': False, 'near_2x': False, 'near_3x': False,
        'rpm': None, 'f_rot': None,
        'top_peaks_hz': [], 'orders': []
    }
    if not isinstance(knowledge, str) or not knowledge:
        return info
    # rpm/f_rot
    m = re.search(r"rpm=([\d\.]+).*?f_rot=([\d\.]+)", knowledge)
    if m:
        info['rpm'] = _parse_float(m.group(1), None)
        info['f_rot'] = _parse_float(m.group(2), None)
    # RMS
    m = re.search(r"RMS\(x,y\)=\(([^,]+),([^\)]+)\)", knowledge)
    if m:
        info['rms_x'] = _parse_float(m.group(1), None)
        info['rms_y'] = _parse_float(m.group(2), None)
    # Kurtosis
    m = re.search(r"Kurtosis\(x,y\)=\(([^,]+),([^\)]+)\)", knowledge)
    if m:
        info['kurt_x'] = _parse_float(m.group(1), None)
        info['kurt_y'] = _parse_float(m.group(2), None)
    # CrestFactor
    m = re.search(r"CrestFactor\(x,y\)=\(([^,]+),([^\)]+)\)", knowledge)
    if m:
        info['cf_x'] = _parse_float(m.group(1), None)
        info['cf_y'] = _parse_float(m.group(2), None)
    # Peaks Hz
    m = re.search(r"TopPeaksHz=([^|]+)", knowledge)
    if m:
        raw = m.group(1)
        peaks = []
        for p in raw.split(','):
            p = p.strip()
            if p:
                peaks.append(_parse_float(p, None))
        info['top_peaks_hz'] = [p for p in peaks if p is not None]
    # Orders
    m = re.search(r"Orders=([^|]+)", knowledge)
    if m:
        raw = m.group(1)
        orders = []
        for o in raw.split(','):
            o = o.strip()
            if o:
                orders.append(_parse_float(o, None))
        info['orders'] = [o for o in orders if o is not None]
    # near flags
    for key in ["near(1x)", "near(2x)", "near(3x)"]:
        m = re.search(fr"{re.escape(key)}=([A-Za-z]+)", knowledge)
        if m:
            val = m.group(1).lower() == 'true'
            if key == "near(1x)":
                info['near_1x'] = val
            elif key == "near(2x)":
                info['near_2x'] = val
            else:
                info['near_3x'] = val
    return info

def _make_queries(cur_info: dict, norm_info: dict) -> List[str]:
    """
    Build retrieval queries from physical indicators.
    Emphasize harmonics (1x/2x/3x), kurtosis (impulsive/bearing), order spectrum, envelope analysis, looseness/misalignment.
    """
    queries = []
    # Base keywords
    queries.append("rotating machinery vibration diagnosis order spectrum harmonics guide")
    # Harmonic cues
    if cur_info.get('near_1x'):
        queries.append("unbalance diagnosis 1x order peak rules")
    if cur_info.get('near_2x'):
        queries.append("misalignment diagnosis 2x order spectrum criteria")
    if cur_info.get('near_3x'):
        queries.append("mechanical issues 3x harmonic diagnosis guidance")
    # Impulsiveness cues
    kx, ky = cur_info.get('kurt_x'), cur_info.get('kurt_y')
    if (kx and kx > 3.5) or (ky and ky > 3.5):
        queries.append("bearing fault diagnosis high kurtosis envelope spectrum BPFO BPFI")
    # Looseness (broadband, multiple harmonics)
    if cur_info.get('orders') and len(cur_info['orders']) >= 3:
        queries.append("mechanical looseness multiple harmonics vibration criteria")
    # Order/Envelope method references
    queries.append("order tracking vibration analysis thresholds practical")
    queries.append("envelope analysis bearing outer inner race criteria")
    return list(dict.fromkeys(queries))  # deduplicate preserving order

def _summarize_docs(docs: List, max_chars: int = 1200) -> str:
    """Rule-based short summarizer for retrieved LangChain Document objects."""
    if not docs:
        return ""
    # Keywords to keep sentences concise and relevant
    KEYWORDS = [
        'unbalance','misalignment','looseness','bearing','outer race','inner race','ball','cage',
        'BPFO','BPFI','BSF','FTF','order','harmonic','1x','2x','3x','envelope','kurtosis','crest','RMS','threshold','criterion','criteria','diagnosis'
    ]
    def score_sentence(s: str) -> int:
        s_lower = s.lower()
        score = sum(1 for kw in KEYWORDS if kw in s_lower)
        if any(ch.isdigit() for ch in s):
            score += 1
        return score
    # Collect sentences from top docs
    sentences = []
    for d in docs:
        txt = d.page_content if hasattr(d, 'page_content') else str(d)
        # naive split
        parts = re.split(r"(?<=[\.!?])\s+", txt)
        for p in parts:
            p = p.strip()
            if 20 <= len(p) <= 300:
                sentences.append(p)
    # Rank and pick
    sentences = sorted(sentences, key=score_sentence, reverse=True)[:20]
    out = []
    total = 0
    for s in sentences:
        if total + len(s) + 1 > max_chars:
            break
        out.append("- " + s)
        total += len(s) + 1
    return "\n".join(out)


from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Union
from accelerate.utils import gather_object
from trl.data_utils import maybe_apply_chat_template
import re
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset, visualize_imaging_tensor

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
    def __init__(self, vibration_dataset, vib_encoder, embedding_dim, retriever=None, max_retrievals: int = 6):
        self.vibration_dataset = vibration_dataset
        self.embedding_dim = int(embedding_dim)
        # vib_encoder may be None (not implemented yet). If provided, build tokenizer; else, defer to zero-vector tokens.
        if vib_encoder is not None:
            self.vib_tokenizer = VibEmbedding(
                vib_encoder=vib_encoder,
                token_embed_dim=self.embedding_dim
            )
        else:
            self.vib_tokenizer = None
        self.retriever = retriever
        self.max_retrievals = int(max_retrievals)

    def __len__(self):
        return len(self.vibration_dataset)

    def __getitem__(self, idx):
        (
            x_img, x_cls, x_info, n_img, n_cls, n_info
        ) = self.vibration_dataset.__getitem__(idx, data_info=True)

        x_knowledge = x_info.get('knowledge', '')
        n_knowledge = n_info.get('knowledge', '')

        # Build vibration tokens (or zeros if vib_encoder is not available yet)
        if self.vib_tokenizer is not None:
            sample_tensor = x_img.unsqueeze(0)
            normal_tensor = n_img.unsqueeze(0)
            current_token, normal_token = self.vib_tokenizer(sample_tensor, normal_tensor)
            current_token = current_token.squeeze(0).cpu()
            normal_token = normal_token.squeeze(0).cpu()
        else:
            current_token = torch.zeros(self.embedding_dim, dtype=torch.float32)
            normal_token  = torch.zeros(self.embedding_dim, dtype=torch.float32)

        # ===== RAG: derive physical info, retrieve, and summarize =====
        cur_info = _extract_physical_info(x_knowledge)
        norm_info = _extract_physical_info(n_knowledge)
        retrieved_summary = ""
        if self.retriever is not None:
            queries = _make_queries(cur_info, norm_info)
            docs_all = []
            try:
                for q in queries[:self.max_retrievals]:
                    docs = self.retriever.get_relevant_documents(q)
                    docs_all.extend(docs)
            except Exception:
                docs_all = []
            retrieved_summary = _summarize_docs(docs_all, max_chars=1200)
        # Pack a compact JSON-like physical summary for the prompt
        phys_summary_lines = []
        for k in ['rpm','f_rot','rms_x','rms_y','kurt_x','kurt_y','cf_x','cf_y','near_1x','near_2x','near_3x']:
            v = cur_info.get(k)
            phys_summary_lines.append(f"{k}={v}")
        phys_summary = ", ".join(phys_summary_lines)

        system_prompt = (
            "You are an expert in vibration-based diagnosis of rotating machinery. "
            "Follow the steps strictly and keep answers concise. Always include a brief rationale."
        )

        # 3-stage prompting: (A) Vibration-only -> (B) Knowledge-only -> (C) Fused decision
        user_prompt = (
            "Possible conditions: looseness, normal, unbalance, misalignment, bearing.\n\n"
            "=== (A) Vibration-only Diagnosis ===\n"
            "Use ONLY the two vibration embeddings below. DO NOT use any external knowledge.\n"
            "Normal state embedding: <NORMAL_VIB_EMB>\n"
            "Current state embedding: <CURRENT_VIB_EMB>\n"
            "Output JSON with fields: {'vib_only_label': <one_of_labels>, 'vib_reason': <one_sentence>}.\n\n"
            "=== (B) Knowledge-only Diagnosis (with RAG) ===\n"
            "Step 1) Physical info parsed from signals (current): " + phys_summary + "\n"
            "Step 2) Retrieved domain knowledge (summarized bullets):\n" + (retrieved_summary if retrieved_summary else "- (no retrieval or empty)") + "\n"
            "Step 3) Using Step 2, derive explicit diagnostic criteria as short bullet rules (thresholds if relevant).\n"
            "Step 4) Apply your criteria to CURRENT vs NORMAL knowledge below to decide the condition.\n"
            f"Normal knowledge: ```{n_knowledge}```\n"
            f"Current knowledge: ```{x_knowledge}```\n"
            "Output JSON with fields: {'knowledge_only_label': <one_of_labels>, 'knowledge_reason': <one_sentence>, 'criteria': <one_or_two_bullets>}.\n\n"
            "=== (C) Fused Decision ===\n"
            "Combine (A) and (B). If they agree, keep the label. If they disagree, prefer the label with stronger evidence; "
            "when uncertain, defer to knowledge-only. Output JSON with fields: "
            "{'final_label': <one_of_labels>, 'fusion_reason': <one_sentence>}."
        )

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        # Supervised target (can be replaced by gold label if available)
        assistant_response = f"The diagnosis result is {x_info['label_class']}."

        return {
            'prompt': prompt_only,
            'answers': assistant_response,
            'normal_token': normal_token,
            'currnet_token': current_token
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

    # vib_encoder is not implemented yet
    vib_encoder = None  # SegmentReconModel()  # not implemented yet

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

    # RAG 시스템 만들기
    # 1. expertise 폴더에 있는 pdf 파일들을 불러오기
    expertise_folder_path = './docs'
    pdf_files = [os.path.join(expertise_folder_path, f) for f in os.listdir(expertise_folder_path) if f.endswith(".pdf")]
    documents = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file)
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 2. Qwen3-embedding-8B 임베딩 모델을 사용하여 문서 임베딩
    embeddings = HuggingFaceEmbeddings(
                                        model_name="Qwen/Qwen3-embedding-8B", 
                                        encode_kwargs={"normalize_embeddings":True}, 
                                        cache_folder="./model_cache"
                                    )

    # 3. VectorDB에 문서 저장
    persist_directory = os.path.join(expertise_folder_path, "vectorstore")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])
    
    # Derive token embedding dimension directly from the LLM's input embedding matrix
    token_embed_dim = int(llm.get_input_embeddings().embedding_dim)

    # Dataset Loading
    data_root = os.path.join(os.getcwd(), 'processed')
    data_mode = 'stft+cross'
    signal_imger = OrderInvariantSignalImager(
        mode=data_mode,
        log1p=True,
        normalize="per_channel",  # None | "per_channel" | "global"
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=20.0,           # order 축 상한
        H_out=256,                # order-bin 수
        W_out=256,                # time-bin 수
        # STFT
        stft_nperseg=1024,
        stft_hop=256,
        stft_window="hann",
        stft_center=True,
        stft_power=1.0,           # 1: magnitude, 2: power
    )

    source_dataset = ['iis', 'vat', 'vbl', 'mfd']
    traget_dataset = ['dxai']
    trainset = WindowedVibrationDataset(
        data_root=data_root,
        using_dataset=source_dataset,
        window_sec=5,
        stride_sec=2,
        cache_mode='file',
        transform=signal_imger
    )
    llm_trainset = VibrationSFTDatasetEnglish(
        vibration_dataset=trainset,
        vib_encoder=vib_encoder,
        embedding_dim=token_embed_dim,
        retriever=retriever,
        max_retrievals=6,
    )
    
    sample = llm_trainset[0]
    print(sample['prompt'])
    print(sample['answers'])


    # valset = WindowedVibrationDataset(
    #     data_root=data_root,
    #     using_dataset=traget_dataset,
    #     window_sec=5,
    #     stride_sec=2,
    #     cache_mode='file',
    #     transform=signal_imger
    # )
    # llm_valset = VibrationSFTDatasetEnglish(
    #     vibration_dataset=valset,
    #     vib_encoder=vib_encoder,
    #     embedding_dim=token_embed_dim
    # )
# 
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
    #     train_dataset=llm_trainset,
    #     eval_dataset=llm_valset,
    #     reward_funcs=[reward_format, reward_accuracy],
    # )

    # if llm_trainset.vib_tokenizer is not None:
    #     for name, param in llm_trainset.vib_tokenizer.vib_encoder.named_parameters():
    #         if param.requires_grad:
    #             print(f"[ENCODER] Trainable: {name}")
    #     for name, param in llm_trainset.vib_tokenizer.model.named_parameters():
    #         if param.requires_grad:
    #             print(f"[VIB_EMBEDDING] Trainable: {name}")
    # else:
    #     print("[INFO] vib_encoder is None; using zero-vectors for vibration tokens.")

    # vib_embedder = llm_trainset.vib_tokenizer
    # model = trainer.model
    # learning_rate = training_args.learning_rate
    # if vib_embedder is not None:
    #     params = list(filter(lambda p: p.requires_grad, vib_embedder.parameters())) + \
    #              list(filter(lambda p: p.requires_grad, model.parameters()))
    # else:
    #     params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # trainer.optimizer = torch.optim.AdamW(params, lr=learning_rate)

    # trainer.train()