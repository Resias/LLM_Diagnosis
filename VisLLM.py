import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from peft import get_peft_model, LoraConfig, TaskType

from models.vit_encoder_recon import VITEnClassify
from VisLLM.vib_grpo import CustomGRPOTRainer, reward_accuracy, reward_format
from VisLLM.vllm_dataset import make_retreiver, Planner, VibrationTokenizer, VibrationSFTDataset
from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    parser.add_argument('--dataset_root',   type=str, default='./parsed', help='Root path to the processed dataset')
    parser.add_argument('--docs_path',      type=str, default='./docs', help='Path to folder containing TXT expertise docs')
    parser.add_argument('--model_cache',    type=str, default='./output', help='Directory to cacheing LLM Models')
    
    parser.add_argument('--llm_model',      type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='LLM Model name')
    parser.add_argument('--embedding_model',type=str, default='Qwen/Qwen3-embedding-0.6B', help='Embedding Model name for RAG')
    parser.add_argument('--planner_max_token', type=int, default=4000, help='Max new tokens for main generation (trainer)')
    args = parser.parse_args()
    
    
    # Tokenizner & LLM & RAG (+special token/LoRA setting)
    print('Loading Tokenizer, LLM ...')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model,
                                            cache_dir=args.model_cache)
    
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.float16,
                                            cache_dir=args.model_cache)
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
    
    vib_encoder = VITEnClassify(
        num_classes=5
    )
    token_embed_dim = int(llm.get_input_embeddings().embedding_dim)
    
    vib_tokenizer = VibrationTokenizer(
        vib_encoder=vib_encoder,
        token_embed_dim=token_embed_dim,
        freeze_encoder=False
    )
    print('Setting RAG ... ')
    retriever = make_retreiver(
        embedding_model=args.embedding_model,
        model_cache=args.model_cache,
        docs_path=args.docs_path,
        retriever_k=15
    )
    
    
    # Planer
    print('Setting Planner ... ')
    planner = Planner(tokenizer=tokenizer,
                        llm=llm,
                        retreiver=retriever,
                        max_tokens=args.planner_max_token)
    
    
    # Dataset
    print('Building Dataset ... ')
    data_root = args.dataset_root
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
    llm_trainset = VibrationSFTDataset(
        vibration_dataset=trainset,
        vibration_tokenizer=vib_tokenizer,
        planner = planner
    )
    valset = WindowedVibrationDataset(
        data_root=data_root,
        using_dataset=traget_dataset,
        window_sec=5,
        stride_sec=2,
        cache_mode='file',
        transform=signal_imger
    )
    llm_valset = VibrationSFTDataset(
        vibration_dataset=valset,
        vibration_tokenizer=vib_tokenizer,
        planner = planner
    )
    
    # Training
    print('Start Training!')
    training_args = GRPOConfig(
        output_dir=args.model_out,
        logging_strategy="steps",
        logging_steps=10,
        report_to="wandb",
    )
    trainer = CustomGRPOTRainer(
        model=llm,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=llm_trainset,
        eval_dataset=llm_valset,
        reward_funcs=[reward_format, reward_accuracy],
    )

    vib_embedder = llm_trainset.vib_tokenizer
    model = trainer.model
    learning_rate = training_args.learning_rate
    params = list(filter(lambda p: p.requires_grad, vib_embedder.parameters())) + \
            list(filter(lambda p: p.requires_grad, model.parameters()))

    trainer.optimizer = torch.optim.AdamW(params, lr=learning_rate)

    trainer.train()