import os
import argparse
import torch
import wandb
from itertools import chain
from torch.optim import AdamW

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from functools import partial

from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE
from tokenizer_trainer.vib_tokenizer import VibrationTokenizer
from SFT.vib_sft_trainer import VibrationSFTTrainer, compute_metrics, preprocess_logits_for_metrics
from SFT.sft_dataset import data_generator, format_and_tokenize_function, VibDataCollatorWrapper
from functools import partial 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    # 데이터셋 관련 옵션들
    parser.add_argument('--data_root',   type=str, default='./llm_dataset.pt', help='llm_dataset_caching.py를 통해 만들어진 데이터 pt파일경로')
    
    # 캐싱 경로 옵션들
    parser.add_argument('--model_cache',    type=str, default='./model_cache', help='LLM 모델들을 caching해둘 경로 (TRANSFORMERS_CACHE)')
    parser.add_argument('--hf_home',        type=str, default='./.hf_home', help='HuggingFace 캐시 루트 (datasets/hub)')
    parser.add_argument('--datasets_cache', type=str, default=None, help='HF_DATASETS_CACHE 경로 (지정 없으면 hf_home/datasets)')
    parser.add_argument('--hub_cache',      type=str, default=None, help='HF_HUB_CACHE 경로 (지정 없으면 hf_home/hub)')
    
    # 학습 결과물 저장 옵션들
    parser.add_argument('--model_out',    type=str, default='./output', help='학습 결과가 저장될 디렉토리')
    parser.add_argument('--log_dir',    type=str, default='./output', help='학습 결과가 저장될 디렉토리')
    
    # LLM 모델 관련 옵션들
    parser.add_argument('--run_name',    type=str, default='0904', help='wandb에 저장될 run 이름')
    parser.add_argument('--llm_model',      type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='LLM Model name')
    args = parser.parse_args()
    
    # HuggingFace가 자꾸 지멋대로 캐싱해서 수정사항이 반영안되는 문제가 있음 -> 캐싱 없애기
    os.makedirs(args.hf_home, exist_ok=True)
    os.environ["HF_HOME"] = os.path.abspath(args.hf_home)
    os.environ["TRANSFORMERS_CACHE"] = os.path.abspath(args.model_cache)
    ds_cache = os.path.abspath(args.datasets_cache) if args.datasets_cache else os.path.join(os.environ["HF_HOME"], "datasets")
    hub_cache = os.path.abspath(args.hub_cache) if args.hub_cache else os.path.join(os.environ["HF_HOME"], "hub")
    os.makedirs(ds_cache, exist_ok=True)
    os.makedirs(hub_cache, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = ds_cache
    os.environ["HF_HUB_CACHE"] = hub_cache
    datasets.disable_caching()

    # 1. Tokenizer, LLM 모델 세팅
    print('Loading Tokenizer, LLM ...')
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model,
                                            cache_dir=args.model_cache)
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model, quantization_config=nf4_config, device_map="auto",
                                            cache_dir=args.model_cache)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)
    llm.print_trainable_parameters()
    special_tokens = {
        'additional_special_tokens': ["<NORMAL_VIB_EMB>", "<CURRENT_VIB_EMB>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    tokenizer.padding_side = "right"
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.resize_token_embeddings(len(tokenizer))
    
    # 2. Vibration Tokenizer 세팅
    vib_ae = VisionTransformerAE(
                                        num_classes=5,
                                    )
    vib_tokenizer = VibrationTokenizer(
                                        vib_ae=vib_ae,
                                        token_embed_dim=int(llm.get_input_embeddings().embedding_dim),
                                        freeze_encoder=True,
                                    )
    
    # 3. Dataset 세팅
    signal_imger = OrderInvariantSignalImager(
                                mode='stft+cross',
                                log1p=True,
                                normalize= "per_channel",  
                                eps=1e-8,
                                out_dtype=torch.float32,
                                max_order=20.0,           
                                H_out=224,                
                                W_out=224,               
                                stft_nperseg=1024,
                                stft_hop=256,
                                stft_window="hann",
                                stft_center=True,
                                stft_power=1.0,           
                            )
    vib_trainset = WindowedVibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['dxai'],
                                window_sec=5,
                                stride_sec=3,
                                cache_mode='none',                      
                                transform=signal_imger,
                                dict_style=True,
                                test_mode=True
                            )
    vib_valset = WindowedVibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['dxai'],
                                window_sec=5,
                                stride_sec=3,
                                cache_mode='none',                      
                                transform=signal_imger,
                                dict_style=True,
                                test_mode=True
                            )
    
    train_hf_dataset = datasets.Dataset.from_generator(
        data_generator, gen_kwargs={"vib_dataset": vib_trainset}, keep_in_memory=True
    )
    eval_hf_dataset = datasets.Dataset.from_generator(
        data_generator, gen_kwargs={"vib_dataset": vib_valset}, keep_in_memory=True
    )

    map_function_with_tokenizer = partial(format_and_tokenize_function, tokenizer=tokenizer)
    tokenized_train_dataset = train_hf_dataset.map(
        map_function_with_tokenizer,
        remove_columns=[c for c in train_hf_dataset.column_names if c in ("merged_class",)],
        load_from_cache_file=False,
    )
    tokenized_eval_dataset = eval_hf_dataset.map(
        map_function_with_tokenizer,
        remove_columns=[c for c in eval_hf_dataset.column_names if c in ("merged_class",)],
        load_from_cache_file=False,
    )
    
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)
    os.environ["WANDB_PROJECT"] = "Vibration-LLM-SFT" 
    os.environ["WANDB_RUN_NAME"] = args.run_name
    training_args = TrainingArguments(
                                    output_dir=args.model_out,             

                                    # --- 학습 하이퍼파라미터 ---
                                    num_train_epochs=200,               # 총 학습 에포크
                                    per_device_train_batch_size=4,      # GPU 당 배치 사이즈
                                    gradient_accumulation_steps=8,      # 그래디언트 축적 스텝 (실질 배치 사이즈: 4 * 8 = 32)
                                    
                                    # --- 옵티마이저 관련 ---
                                    learning_rate=2e-5,                 # 학습률
                                    weight_decay=0.01,                  # 가중치 감쇠
                                    
                                    # --- 로깅 및 저장 ---
                                    logging_dir=args.log_dir,               # 로그 저장 경로
                                    logging_strategy='epoch',
                                    save_strategy="epoch",        
                                    report_to="wandb",      
                
                                    # --- 평가 ---
                                    eval_strategy="epoch",      
    
                                    # --- 동작관련 옵션들 ---
                                    remove_unused_columns=False,
                                    label_smoothing_factor=0.1 
                                )


        
    # model과 vib_tokenizer의 파라미터를 하나로 합치기
    combined_parameters = chain(llm.parameters(), vib_tokenizer.alignment_layer.parameters())
    optimizer = AdamW(combined_parameters)
    
    
    trainer = VibrationSFTTrainer(
                                    model = llm,
                                    processing_class = tokenizer,
                                    train_dataset = tokenized_train_dataset,
                                    eval_dataset = tokenized_eval_dataset,
                                    args=training_args,
                                    optimizers = (optimizer, None),
                                    vib_tokenizer=vib_tokenizer,
                                    compute_metrics=compute_metrics_with_tokenizer,
                                    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                    
                                )
    trainer.data_collator = VibDataCollatorWrapper(
        original_collator=trainer.data_collator
    )
    trainer.train()
    wandb.finish()
