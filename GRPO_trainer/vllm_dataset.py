import torch
import torch.nn as nn
import os
import time
import shutil
from torch.utils.data import Dataset
import torch.distributed as dist

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document



class Planner:
    def __init__(self,
                tokenizer,
                llm,
                retreiver,
                max_tokens,
                device
                ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.retreiver = retreiver
        self.max_tokens = max_tokens
        self.device = device
        
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
            input_ids.to(self.device)
            out_ids = self.llm.generate(**input_ids, max_new_tokens=self.max_tokens, do_sample=False)
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return out_text.strip()
    
    def __call__(self, current_knowledge, normal_knowledge):
        
        retrive_docs = self.retreive(current_knowledge, normal_knowledge)
        plan = self.plan(current_knowledge, normal_knowledge, retrive_docs)
        
        evidence = retrive_docs
        json_part = plan.split('Assistant: Output JSON only, no extra text. Do not hallucinate.')[1]
        
        return json_part

class VibrationTokenizer(nn.Module):
    def __init__(self, vib_encoder, token_embed_dim, freeze_encoder=True, embedding_dim=768):
        super().__init__()
        self.vib_encoder = vib_encoder
        self.device = next(self.vib_encoder.parameters()).device
        self.dtype = next(self.vib_encoder.parameters()).dtype

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

    def forward(self, x):
        # Ensure inputs are on the same device as the encoder/model
        device = next(self.vib_encoder.parameters()).device if self.vib_encoder is not None else next(self.model.parameters()).device

        current_tensor = x.unsqueeze(0).to(device)

        sample_attn = self.vib_encoder._encode_full(current_tensor)
        sample_attn = sample_attn[:, 0, :]  # 768

        z = self.model(sample_attn)

        # Return CPU tensors so DataLoader pin_memory works properly
        return z.detach().cpu()

class VibrationSFTDataset(Dataset):
    def __init__(self,
                vibration_dataset,
                vib_tokenizer,
                planner,
                device,
                test_mode=True):
        self.vibration_dataset = vibration_dataset
        self.vib_tokenizer = vib_tokenizer
        self.planner = planner
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
        self.device=device 
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.vibration_dataset)
    
    def __getitem__(self, idx):
        (
            current_x, _, current_info, normal_x, _, normal_info
        ) = self.vibration_dataset.__getitem__(idx, data_info=True)
        # Keep tensors on CPU; tokenizer will move to correct device internally

        current_knowledge = current_info['knowledge']
        normal_knowledge = normal_info['knowledge']
        
        
        current_token, normal_token = self.vib_tokenizer(current_x, normal_x)
        current_token = current_token.squeeze(0)
        normal_token = normal_token.squeeze(0)
        
        if self.test_mode:
            print('Skip Plan')
            plan_text = 'See you Later'
        else:
            plan_text = self.planner(current_knowledge, normal_knowledge)
        
        system_prompt = (
            f"""You are a domain expert in vibration-based rotating machinery diagnostics.
            Make precise diagnosis classification among {self.target_labels} from User question.
            """
        )
        print(plan_text)
        exit()
        # 요구사항: reasoning(접근 계획/추론)과 result(최종 JSON) 분리 출력
        # - reasoning: 단계적으로 사고 과정을 요약 (Step 1, Step 2 ...)
        # - answer: 기존 JSON 형식 하나만 출력
        user_prompt = (
            f"Possible conditions: {self.target_labels}.\n\n"
            "You will perform a 3-stage diagnostic reasoning process comparing NORMAL vs CURRENT states.\n"
            "Use ONLY the two vibration embeddings when doing vibration-only reasoning"
            "Use provided knowledge when doing knwoledge-only reasoning.\n"
            "Do not include any code fences. Output exactly one reasoning block and one answer block.\n\n"
            "Vibration embeddings:\n"
            "- Normal state embedding: <NORMAL_VIB_EMB>\n"
            "- Current state embedding: <CURRENT_VIB_EMB>\n\n"
            f"- Normal state feature: {current_knowledge}\n"
            f"- Normal state feature: {normal_knowledge}\n"
            f"Diagnosis plan :\n{plan_text}\n\n"
            "Follow the EXACT structure below. Write detailed analysis in <reasoning> and concise result in <answer>."
            """
            <reasoning>
            MAIN STEP 1 — Embedding-based Comparison
            1.1 Compute or conceptually assess similarity/distance between <CURRENT_VIB_EMB> and <NORMAL_VIB_EMB> (e.g., cosine proximity). 
            1.2 Identify which label prototypes the CURRENT embedding is closest to, if implied by features/notes.
            1.3 Summarize whether embeddings alone suggest normality or a specific fault class, and why.

            MAIN STEP 2 — Feature- & Knowledge-based Classification
            2.1 Compare CURRENT vs NORMAL for RMS, kurtosis, crest factor per axis. Note significant deviations.
            2.2 Inspect spectral content in orders: 
                - Check prominence at 1×, 2×, 3× and higher-order harmonics.
                - Note sidebands, modulations, or resonance clusters.
                - Map observed peaks to bearing fault frequencies (BPFO/BPFI/BSF/FTF) when provided.
            2.3 Cross-reference CurrentKnowledgeSnippets and NormalKnowledgeSnippets with observed stats/spectrum:
                - Confirm or refute each snippet with concrete evidence.
            2.4 Derive a label from features+knowledge with a rationale and uncertainty.

            MAIN STEP 3 — Fused Decision
            3.1 Reconcile STEP 1 and STEP 2 outcomes. If they agree, keep the label; if not, pick the label with stronger evidence and explain why.
            3.2 State key indicators (top 2–3) that most strongly support the final label.
            3.3 Provide a calibrated confidence (0–1), reflecting agreement between steps, signal quality, and evidence strength.
            </reasoning>
            """
            "<answer>{\n"
            "  \"vib_only_label\": <one_of_labels>,\n"
            "  \"vib_reason\": <one_sentence>,\n"
            "  \"knowledge_only_label\": <one_of_labels>,\n"
            "  \"knowledge_reason\": <one_sentence>,\n"
            "  \"criteria\": <one_or_two_bullets>,\n"
            "  \"final_label\": <one_of_labels>,\n"
            "  \"fusion_reason\": <one_sentence>\n"
            "}</answer>\n"
            "Constraints: Only one reasoning block and one answer block. No extra text."
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


class LLMDataset_Cache(torch.utils.data.Dataset):
    """
    캐시를 원본 튜플 형식으로 다시 내보내는 래퍼
    (current_x, _, current_info, normal_x, _, normal_info, plan_text)
    """
    def __init__(self, cache_blob_path, using_dataset=['iis']):
        cache_blob = torch.load(cache_blob_path)
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
        current_x = r["current_x"]
        current_info = r["current_info"]
        normal_x = r["normal_x"]
        normal_info = r["normal_info"]
        plan_text = r["plan_text"]
        
        current_knowledge = current_info['knowledge']
        normal_knowledge = normal_info['knowledge']
        
        system_prompt = (
            "You are a world-class AI diagnostic engineer specializing in the vibration analysis of rotating machinery. "
            "Your mission is to meticulously analyze vibration data and associated physical knowledge to deliver a precise and well-supported diagnosis. "
            "You must follow a structured, multi-stage reasoning process and present your findings in the exact format required."
        )
        user_prompt = f"""
        ### **TASK DESCRIPTION**
        Your task is to diagnose the current state of a rotating machine. Based on the provided data, you must classify the state into one of the following five categories: **normal(healthy), misalignment, looseness, unbalance, bearing fault**.

        ### **PROVIDED DATA**

        **1. Vibration Tokens (Proprietary Embeddings):**
        - Normal State Token: `<NORMAL_VIB_EMB>`
        - Current State Token: `<CURRENT_VIB_EMB>`

        **2. Physical Knowledge (Extracted Features):**
        - Normal State Features: "{normal_knowledge}"
        - Current State Features: "{current_knowledge}"

        **3. Analysis Plan & Diagnostic Criteria (Reference Guide):**
        This guide outlines the step-by-step analysis process and the criteria for each potential fault. You must use this as your primary reference for knowledge-based analysis.
        - Plan: {plan_text}

        ### **INSTRUCTIONS**

        You must perform a rigorous 3-step diagnostic reasoning process. Adhere strictly to the following structure within the `<reasoning>` block.

        <reasoning>
        **Step 1: Vibration Token-based Analysis**
        1.1. Compare the `<CURRENT_VIB_EMB>` with the `<NORMAL_VIB_EMB>`. Based on their similarity (or lack thereof), what is the initial diagnosis? A significant deviation from the normal token suggests a fault.
        1.2. Briefly state the most likely condition suggested by the vibration tokens alone.

        **Step 2: Physical Knowledge-based Analysis**
        2.1. **Feature Comparison**: Compare the 'Current State Features' against the 'Normal State Features'. Note any significant differences in RMS, Kurtosis, and Crest Factor.
        2.2. **Spectral Analysis**: Examine the 'TopPeaksHz' and 'Orders' in the current state. According to the 'Analysis Plan', do these spectral components indicate a specific fault (e.g., harmonics for unbalance/misalignment, non-harmonics for bearing faults)?
        2.3. **Criteria Matching**: Systematically check the diagnostic ideas for each fault type listed in the 'Analysis Plan'. Which condition's criteria best match the 'Current State Features'? Provide evidence.

        **Step 3: Fused Decision**
        3.1. **Synthesize Findings**: Combine the conclusions from Step 1 (token-based) and Step 2 (knowledge-based).
        3.2. **Final Diagnosis**: If they agree, confirm the diagnosis. If they conflict, determine which conclusion is supported by stronger evidence and state your final diagnosis.
        3.3. **Key Indicators**: List the top 2-3 most critical indicators from the physical features that led to your final decision.
        </reasoning>

        ### **OUTPUT FORMAT**

        Provide your final answer in the JSON format below, enclosed within an `<answer>` block. Do not add any text or explanations outside the `<reasoning>` and `<answer>` blocks.

        <answer>{{
        "vib_only_label": "<The diagnosis from Step 1.2>",
        "vib_reason": "<A brief sentence explaining the token-based conclusion>",
        "knowledge_only_label": "<The diagnosis from Step 2.3>",
        "knowledge_reason": "<A brief sentence explaining the knowledge-based conclusion>",
        "criteria": "<The key indicators from Step 3.3>",
        "final_label": "<Your final diagnosis from Step 3.2>",
        "fusion_reason": "<A brief sentence explaining how you reconciled the two analyses to reach the final decision>"
        }}</answer>
        """

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        assistant_response = current_info['merged_class']
        if assistant_response == 'normal':
            assistant_response = 'normal(healthy)'
        elif assistant_response == 'bearing':
            assistant_response = 'bearing fault'

        return {
            'prompt': prompt_only,
            'answers': assistant_response,
            'current_x': current_x,
            'normal_x': normal_x
        }