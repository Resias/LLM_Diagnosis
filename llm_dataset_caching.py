from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset
import torch

from tqdm import tqdm
import os
import time
import shutil
import torch.distributed as dist
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM

class Planner:
    def __init__(self,
                tokenizer,
                llm,
                retreiver,
                max_tokens,
                generation_kwargs,
                device
                ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.retreiver = retreiver
        self.max_tokens = max_tokens
        self.device = device
        self.generation_kwargs = generation_kwargs
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
            f"Evidence: {retrive_docs} \n"
            "Assistant: Output JSON only, no extra text."
        )
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors='pt')
            input_ids.to(self.device)
            out_ids = self.llm.generate(**input_ids, 
                                        do_sample=True,
                                        max_new_tokens=self.max_tokens,
                                        temperature = self.generation_kwargs['temperature'],
                                        top_k = self.generation_kwargs['top_k'],
                                        top_p = self.generation_kwargs['top_p'],
            )
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return out_text.strip()
    
    def __call__(self, current_knowledge, normal_knowledge):
        
        retrive_docs = self.retreive(current_knowledge, normal_knowledge)
        plan = self.plan(current_knowledge, normal_knowledge, retrive_docs)
        
        evidence = retrive_docs
        json_part = plan.split("Assistant: Output JSON only, no extra text.")[1]
        
        return json_part
    
def make_retreiver(embedding_model, model_cache, docs_path, retriever_k):
    # 1. docs_path 폴더에 있는 TXT 파일들을 불러오기
    txt_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.lower().endswith('.txt')]
    raw_docs = []
    
    rank = int(os.environ.get('RANK', '0'))
    
    is_dist = dist.is_available() and dist.is_initialized()
    
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
    if is_dist:
        if rank == 0:
            print(f'rank : {rank} - Making VectorDB')
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
            dist.barrier()
        else:
            timeout_s = 600
            waited = 0
            print(f'rank : {rank} - Waiting for making VectorDB')
            while not os.path.exists(persist_directory) and waited < timeout_s:
                time.sleep(1)
                waited +=1
            if not os.path.exists(persist_directory):
                raise RuntimeError(f'Time out waiting for vectorstore at {persist_directory}')
            print(f'rank : {rank} - Loading VectorDB')
            vectorstore = Chroma(persist_directory=persist_directory)
    else:
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
            
    
    # 4. Retriever 로 만들어 사용
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    
    return retriever

class LLMDataset(torch.utils.data.Dataset):
    """
    캐시를 원본 튜플 형식으로 다시 내보내는 래퍼
    (current_x, _, current_info, normal_x, _, normal_info, plan_text)
    """
    def __init__(self, cache_blob, using_dataset=['iis']):
        all_records = cache_blob["records"]
        all_datasets = cache_blob["dataset"]
        # using_dataset 안에 포함된 dataset만 남김
        filtered_records = []
        for rec, ds_name in zip(all_records, all_datasets):
            if ds_name in using_dataset:
                filtered_records.append(rec)
        self.records = filtered_records
        self.dataset_names = using_dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        # 원래 구조에 맞춰 반환(빈 placeholder는 None으로)
        return (
            r["current_x"],
            r["current_info"],
            r["normal_x"],
            r["normal_info"],
            r["plan_text"],  # plan_text까지 함께 꺼내 쓰려면 뒤에 붙여서 쓰자
        )

if __name__ == '__main__':
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Set device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print(f'device : {device}')
    data_root = './processed'
    data_mode = 'stft+cross'
    signal_imger = OrderInvariantSignalImager(
        mode=data_mode,
        log1p=True,
        normalize="per_channel",  # None | "per_channel" | "global"
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
    dataset_names = ['dxai', 'iis', 'vat', 'vbl', 'mfd']
    total_dataset = WindowedVibrationDataset(
        data_root=data_root,
        using_dataset=dataset_names,
        window_sec=5,
        stride_sec=2,
        cache_mode='file',
        transform=signal_imger
    )
    
    model_cache = 'model_cache'
    docs_path = 'docs_eng'
    print('Loading LLM and tokenizer')
    llm_model = 'Qwen/Qwen3-4B-Instruct-2507'
    tokenizer = AutoTokenizer.from_pretrained(llm_model,
                                            cache_dir='model_cache')
        
    llm = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16,
                                            cache_dir='model_cache')
    llm.to(device)


    print('Building RAG')
    embedding_model = 'Qwen/Qwen3-embedding-0.6B'
    retriever = make_retreiver(
            embedding_model=embedding_model,
            model_cache=model_cache,
            docs_path=docs_path,
            retriever_k=15
        )
    
    planner_max_token=2000
    generation_kwargs={
                "temperature": 0.15,
                "top_p": 0.9,
                "top_k": 30
            }
    planner = Planner(tokenizer=tokenizer,
                        llm=llm,
                        retreiver=retriever,
                        max_tokens=planner_max_token,
                        generation_kwargs=generation_kwargs,
                        device=device)
    
    records = []
    dataset_records = []
    print('Start Caching...')
    for current_x, _, current_info, normal_x, _, normal_info in tqdm(total_dataset):
        
        current_knowledge = current_info['knowledge']
        normal_knowledge = normal_info['knowledge']
        
        plan_text = planner(current_knowledge, normal_knowledge)
        dataset_records.append(current_info['dataset'])
        rec = {
                "current_x": current_x.detach().cpu(),
                "current_info": current_info,     # dict는 그대로 저장
                "normal_x": normal_x.detach().cpu(),
                "normal_info": normal_info,
                "plan_text": plan_text,
            }
        records.append(rec)
        
    meta = {
            "num_records": len(records),
            "cache_version": 1,
    }
    blob = {"meta": meta, "records": records, 'dataset':dataset_records}
    torch.save(blob, 'llm_dataset.pt')