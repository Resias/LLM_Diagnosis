import argparse
import torch
import torch.distributed as dist
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import wandb

from models.vit_encoder_recon import VITEnClassify
from GRPO_trainer.vib_grpo import CustomGRPOTRainer, reward_accuracy, reward_format
from GRPO_trainer.vllm_dataset import VibrationTokenizer, LLMDataset_Cache

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    parser.add_argument('--data_cache_path',   type=str, default='./llm_dataset.pt', help='llm_dataset_caching.py를 통해 만들어진 데이터 pt파일경로')
    parser.add_argument('--model_cache',    type=str, default='./model_cache', help='LLM 모델들을 caching해둘 경로')
    parser.add_argument('--model_out',    type=str, default='./output', help='학습 결과가 저장될 디렉토리')
    parser.add_argument('--run_name',    type=str, default='0904', help='wandb에 저장될 run 이름')

    parser.add_argument("--distributed", action="store_true", help="Use DDP via torchrun")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="DDP backend")
    parser.add_argument("--dist_url", type=str, default="env://", help="DDP init method (env:// for torchrun)")
    
    parser.add_argument('--llm_model',      type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='LLM Model name')
    parser.add_argument('--embedding_model',type=str, default='Qwen/Qwen3-embedding-0.6B', help='Embedding Model name for RAG')

    parser.add_argument('--max_completion_length', type=int, default=1500, help='Max new tokens for main generation (trainer)')
    args = parser.parse_args()
    
    # --------------- Distributed Setup ---------------
    # Expect to be launched with: torchrun --nproc_per_node=NUM /workspace/VisLLM.py --distributed ...
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = args.distributed and world_size > 1

    # Set device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print(f'device : {device}')
    # Init process group if distributed
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=world_size, rank=rank)
    
    
    # Tokenizner & LLM & RAG (+special token/LoRA setting)
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
    # llm.to(device)
    print(f'llm : {type(llm)}')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)
    special_tokens = {
        'additional_special_tokens': ["<NORMAL_VIB_EMB>", "<CURRENT_VIB_EMB>", "</answer>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    llm.resize_token_embeddings(len(tokenizer))
    
    vib_encoder = VITEnClassify(
        num_classes=5,
        image_size=224
    )
    model_state_dict = torch.load('./best_model.pth')
    vib_encoder.load_state_dict(model_state_dict['model_state_dict'])
    vib_encoder = vib_encoder.to(device)
    token_embed_dim = int(llm.get_input_embeddings().embedding_dim)
    
    vib_tokenizer = VibrationTokenizer(
        vib_encoder=vib_encoder,
        token_embed_dim=token_embed_dim,
        freeze_encoder=False
    ).to(device)
    
    # Dataset
    print('Building Dataset ... ')
    data_cache_path = args.data_cache_path
    source_dataset = ['dxai', 'vat', 'vbl', 'mfd']
    traget_dataset = ['dxai']

    llm_trainset = LLMDataset_Cache(cache_blob_path=data_cache_path,
                                    using_dataset=source_dataset)

    llm_valset = LLMDataset_Cache(cache_blob_path=data_cache_path,
                                    using_dataset=traget_dataset)
    
    # Training
    print('Start Training!')
    if rank==0:
        wandb.init(
            project='VibLLM',
            dir=args.model_out,
            name=args.run_name
        )
    training_args = GRPOConfig(
        output_dir=args.model_out,
        save_strategy="steps",       # 'steps' 기준으로 저장
        save_steps=100,              # 500 스텝마다 저장
        save_total_limit=3,          # 최대 3개의 체크포인트만 유지
        logging_strategy="steps",
        logging_steps=10,
        num_generations=2,
        generation_batch_size=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        num_iterations=10000,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
        report_to="wandb" if rank==0 else 'none',
        generation_kwargs={
            "max_new_tokens": args.max_completion_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20
        }
    )
    vib_embedder = vib_tokenizer.model
    model = llm
    learning_rate = training_args.learning_rate
    params = list(filter(lambda p: p.requires_grad, vib_embedder.parameters())) + \
            list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    
    trainer = CustomGRPOTRainer(
        model=llm,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=llm_trainset,
        eval_dataset=llm_valset,
        reward_funcs=[reward_format, reward_accuracy],
        vib_tokenizer=vib_tokenizer,
        optimizers=(optimizer, None)
    )

    trainer.train()
