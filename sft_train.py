import os
import argparse
import torch
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE
from tokenizer_trainer.vib_tokenizer import VibrationTokenizer
from SFT.vib_sft import VibrationSFTTrainer
from SFT.sft_dataset import VibrationSFT_Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    parser.add_argument('--data_root',   type=str, default='./llm_dataset.pt', help='llm_dataset_caching.py를 통해 만들어진 데이터 pt파일경로')
    parser.add_argument('--model_cache',    type=str, default='./model_cache', help='LLM 모델들을 caching해둘 경로')
    parser.add_argument('--model_out',    type=str, default='./output', help='학습 결과가 저장될 디렉토리')
    parser.add_argument('--run_name',    type=str, default='0904', help='wandb에 저장될 run 이름')
    parser.add_argument('--llm_model',      type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='LLM Model name')
    parser.add_argument('--max_completion_length', type=int, default=1500, help='Max new tokens for main generation (trainer)')
    parser.add_argument('--run_name',    type=str, default='0911_Qwen3-4B', help='wandb에 저장될 run 이름') # <-- run_name 예시 수정
    args = parser.parse_args()
    
    
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
    special_tokens = {
        'additional_special_tokens': ["<NORMAL_VIB_EMB>", "<CURRENT_VIB_EMB>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    llm.resize_token_embeddings(len(tokenizer))
    
    
    # 2. Vibration Tokenizer 세팅
    vib_encoder = VisionTransformerAE(
                                        num_classes=5,
                                    )
    model_state_dict = torch.load('./best_model.pth')
    vib_encoder.load_state_dict(model_state_dict['model_state_dict'])
    token_embed_dim = int(llm.get_input_embeddings().embedding_dim)
    
    vib_tokenizer = VibrationTokenizer(
                                        vib_encoder=vib_encoder,
                                        token_embed_dim=token_embed_dim,
                                        freeze_encoder=True
                                    )
    
    # 3. Dataset 세팅
    signal_imger = OrderInvariantSignalImager(
                                mode='stft+cross',
                                log1p=True,
                                normalize= "per_channel",  
                                eps=1e-8,
                                out_dtype=torch.float32,
                                max_order=20.0,           # order 축 상한
                                H_out=224,                # order-bin 수
                                W_out=224,                # time-bin 수
                                # STFT
                                stft_nperseg=1024,
                                stft_hop=256,
                                stft_window="hann",
                                stft_center=True,
                                stft_power=1.0,           # 1: magnitude, 2: power
                            )
    vib_trainset = WindowedVibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['dxai'],
                                window_sec=5,
                                stride_sec=3,
                                cache_mode='none',                      # file or none
                                transform=signal_imger
                            )
    vib_valset = WindowedVibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['dxai'],
                                window_sec=5,
                                stride_sec=3,
                                cache_mode='none',                      # file or none
                                transform=signal_imger
                            )
    sft_trainset = VibrationSFT_Dataset(
                                    vib_dataset=vib_trainset
                                )
    sft_valset = VibrationSFT_Dataset(
                                    vib_dataset=vib_valset
                                )
    
    os.environ["WANDB_PROJECT"] = "Vibration-LLM-SFT" 
    os.environ["WANDB_RUN_NAME"] = args.run_name
    training_args = TrainingArguments(
                                    # --- 필수 경로 ---
                                    output_dir="./results",             # 모델과 결과물이 저장될 경로

                                    # --- 학습 하이퍼파라미터 ---
                                    num_train_epochs=20,                 # 총 학습 에포크
                                    per_device_train_batch_size=4,      # GPU 당 배치 사이즈
                                    gradient_accumulation_steps=8,      # 그래디언트 축적 스텝 (실질 배치 사이즈: 4 * 8 = 32)
                                    
                                    # --- 옵티마이저 관련 ---
                                    learning_rate=2e-5,                 # 학습률
                                    weight_decay=0.01,                  # 가중치 감쇠
                                    
                                    # --- 로깅 및 저장 ---
                                    logging_dir="./logs",               # 로그 저장 경로
                                    logging_steps=10,                   # 10 스텝마다 로그 출력
                                    save_strategy="epoch",              # 에포크 단위로 모델 저장
                                    
                                    # --- 평가 ---
                                    evaluation_strategy="steps",      # 스텝 단위로 평가 (eval_dataset이 있을 경우)
                                    eval_steps=100,
                                    
                                    report_to="wandb"
                                    
                                )
    trainer = VibrationSFTTrainer(
                                    model=llm,
                                    tokenizer=tokenizer,
                                    train_dataset=sft_trainset,
                                    eval_dataset = sft_valset,
                                    args=training_args,
                                    vib_tokenizer=vib_tokenizer, # 커스텀 인자 전달
                                )
    trainer.train()
    wandb.finish()