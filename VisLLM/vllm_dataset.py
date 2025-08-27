import torch
import torch.nn as nn
import os
import shutil
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset

class Planner:
    def __init__(self,
                tokenizer,
                llm,
                retreiver,
                max_tokens,
                ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.retreiver = retreiver
        self.max_tokens = max_tokens
        
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
        
    def retreive(self, current_knowledge, normal_knowledge):
        
        query = (
            "We have vibration features for rotating machinery. Using ALL provided information, "
            f"explain diagnostic methods to classify among {self.target_labels}. "
            "Provide order-spectrum/envelope/kurtosis-based criteria and practical thresholds if known. "
            f"CURRENT: {current_knowledge}. NORMAL: {normal_knowledge}."
        )
        retrive_docs = self.retreiver.invoke(query)
        
        return retrive_docs
    
    def plan(self, current_knowledge, normal_knowledge, retrive_docs):
        prompt = (
            "System: You are a senior vibration analyst. Be precise and cite sources.\n"
            f"User: We will diagnose rotating machinery among: {self.target_labels}.\n"
            f"Current physical info: {current_knowledge}\n"
            f"Normal physical info: {normal_knowledge}\n"
            "Evidence snippets from manuals/papers are given below, each prefixed with [DOC#]. \n"
            "Extract concrete, actionable rules (thresholds, patterns, symptom descriptions) with citations like [DOC3].\n"
            "Return STRICT JSON with keys: plan_steps, diagnosis_plan.\n"
            "- plan_steps: 5-8 short imperative steps (strings).\n"
            f"- diagnosis_plan: object with keys {self.target_labels}, each a list of diagnosis idea;\n"
            "  every rule item must be a JSON object: {\"diagnosis idea\": <one line>, \"why\": <short reason>, \"source\": \"DOC#\"}.\n"
            "Constraints: Use only information supported by the snippets.\n"
            "Evidence:\n" + (retrive_docs) + "\n"
            "Assistant: Output JSON only, no extra text."
        )
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors='pt')
            out_ids = self.llm.generate(**input_ids, max_new_tokens=self.max_tokens, do_sample=False)
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return out_text.strip()
    
    def summerize(self, plan):
        prompt = (
            "System: You are a senior vibration analyst. Create a compact briefing from the given thinking/plan.\n"
            "User: Summarize given PLAN into STRICT JSON with keys: plan_steps, diagnosis_plan.\n"
            "PLAN:\n" + plan + "\n"
            "Assistant: Output JSON only."
        )
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors='pt')
            out_ids = self.llm.generate(**input_ids, max_new_tokens=self.max_tokens, do_sample=False)
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return out_text.strip()
    
    def __call__(self, current_knowledge, normal_knowledge):
        retrive_docs = self.retreive(current_knowledge, normal_knowledge)
        plan = self.plan(current_knowledge, normal_knowledge, retrive_docs)
        plan_summerize = self.summerize(plan)
        
        return plan_summerize

class VibrationTokenizer(nn.Module):
    def __init__(self, vib_encoder, token_embed_dim, freeze_encoder=True, embedding_dim=256):
        super().__init__()
        self.vib_encoder = vib_encoder

        if freeze_encoder:
            for param in self.vib_encoder.parameters():
                param.requires_grad = False

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=int(embedding_dim*2)
            ),
            nn.Sigmoid(),
            nn.Linear(
                in_features=int(embedding_dim*2),
                out_features=token_embed_dim
            )
        )

    def forward(self, current_x, normal_x):
        current_tensor = current_x.unsqueeze(0)
        normal_tensor = normal_x.unsqueeze(0)
        
        sample_attn = self.vib_encoder._encode_full(current_tensor)
        sample_attn = sample_attn[:, 0, :] # 768
        
        normal_attn = self.vib_encoder._encode_full(normal_tensor)
        normal_attn = normal_attn[:, 0, :] # 768
        
        current_z = self.model(sample_attn)
        normal_z = self.model(normal_attn)
        return current_z, normal_z

class VibrationSFTDataset(Dataset):
    def __init__(self,
                vibration_dataset,
                vibration_tokenizer,
                planner):
        self.vibration_dataset = vibration_dataset
        self.vibration_tokenizer = vibration_tokenizer
        self.planner = planner
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"

    def __len__(self):
        return len(self.vibration_dataset)
    
    def __getitem__(self, idx):
        (
            current_x, _, current_info, normal_x, _, normal_info
        ) = self.vibration_dataset.__getitem__(idx, data_info=True)
        
        current_knowledge = current_info['knowledge']
        normal_knowledge = normal_info['knowledge']
        
        
        current_token, normal_token = self.vib_tokenizer(current_x, normal_x)
        current_token = current_token.squeeze(0)
        normal_token = normal_token.squeeze(0)
        
        plan_text = self.planner(current_knowledge, normal_knowledge)
        
        system_prompt = (
            "You are an expert in vibration-based diagnosis of rotating machinery. "
            "Follow the steps strictly and keep answers concise. Always include a brief rationale."
        )

        # 3-stage prompting: (A) Vibration-only -> (B) Knowledge-only -> (C) Fused decision
        user_prompt = (
            f"Possible conditions: {self.target_labels}.\n\n"
            "=== (A) Vibration-only Diagnosis ===\n"
            "Use ONLY the two vibration embeddings below. DO NOT use any external knowledge.\n"
            "Normal state embedding: <NORMAL_VIB_EMB>\n"
            "Current state embedding: <CURRENT_VIB_EMB>\n"
            "Output JSON with fields: {'vib_only_label': <one_of_labels>, 'vib_reason': <one_sentence>}.\n\n"
            "=== (B) Knowledge-only Diagnosis (with RAG) ===\n"
            f"Step 1) Physical info from signals (current): {current_knowledge}\n"
            f"Physical info from signals (normal): {normal_knowledge}\n"
            f"Step 2) Plan for diagnosis current signal:\n {plan_text}"
            "Step 3) FOLLOW the cited plan in Step 2 to derive explicit diagnostic rules you will apply now (you may restate them briefly).\n"
            "Step 4) Apply your criteria to CURRENT vs NORMAL Physical info to decide the condition.\n"
            "Output JSON with fields: {'knowledge_only_label': <one_of_labels>, 'knowledge_reason': <one_sentence>, 'criteria': <one_or_two_bullets>}.\n\n"
            "=== (C) Fused Decision ===\n"
            "Combine (A) and (B). If they agree, keep the label. If they disagree, prefer the label with stronger evidence; "
            "when uncertain, defer to knowledge-only. Output JSON with fields: "
            "{'final_label': <one_of_labels>, 'fusion_reason': <one_sentence>}."
        )
        
        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        assistant_response = f"The diagnosis result is {current_info['label_class']}."

        return {
            'prompt': prompt_only,
            'answers': assistant_response,
            'normal_token': normal_token,
            'currnet_token': current_token
        }

def make_retreiver(embedding_model, model_cache, docs_path, retriever_k):
    # 1. docs_path 폴더에 있는 TXT 파일들을 불러오기
    txt_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.lower().endswith('.txt')]
    raw_docs = []
    for path in txt_files:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            raw_docs.append(Document(page_content=text, 
                            metadata={'source': os.path.basename(path)}))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_docs)

    # 2. 문서 임베딩
    embeddings = HuggingFaceEmbeddings(
                                        model_name=embedding_model,
                                        encode_kwargs={"normalize_embeddings":True},
                                        cache_folder=model_cache
                                    )

    # 3. VectorDB에 문서 저장
    persist_directory = os.path.join(docs_path, "vectorstore")
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    
    # 4. Retriever 로 만들어 사용
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    
    return retriever
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    parser.add_argument('--dataset_root',   type=str, default='./parsed', help='Root path to the processed dataset')
    parser.add_argument('--docs_path',      type=str, default='./docs', help='Path to folder containing TXT expertise docs')
    parser.add_argument('--model_cache',    type=str, default='./output', help='Directory to cacheing LLM Models')
    
    parser.add_argument('--llm_model',      type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='LLM Model name')
    parser.add_argument('--embedding_model',type=str, default='Qwen/Qwen3-embedding-0.6B', help='Embedding Model name for RAG')
    parser.add_argument('--palnner_max_token', type=int, default=4000, help='Max new tokens for main generation (trainer)')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model,
                                            cache_dir=args.model_cache)
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.float16,
                                            cache_dir=args.model_cache)
    retriever = make_retreiver(
        embedding_model=args.embedding_model,
        model_cache=args.model_cache,
        docs_path=args.docs_path,
        retriever_k=15
    )
    
    
    # Planer
    planner = Planner(tokenizer=tokenizer,
                        llm=llm,
                        retreiver=retriever,
                        max_tokens=args.planner_max_token)
    
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
    
    
    vib_encoder = None # 승하가 구현중
    token_embed_dim = int(llm.get_input_embeddings().embedding_dim)
    vib_tokenizer = VibrationTokenizer(
        vib_encoder=vib_encoder,
        token_embed_dim=int(llm.get_input_embeddings().embedding_dim),
        freeze_encoder=False
    )
    llm_dataset = VibrationSFTDataset(
        vibration_dataset=trainset,
        vibration_tokenizer=vib_tokenizer,
        planner = planner
    )