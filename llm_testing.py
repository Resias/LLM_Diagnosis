import torch
import os
from torch.utils.data import Dataset
from data.order_dataset import ExtendedOrderFreqDataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from trl import GRPOTrainer, GRPOConfig
from utils.reward import reward_format, reward_accuracy
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Dataset
class VibrationSFTDatasetEnglish(Dataset):
    def __init__(self, vibration_dataset, pdf_folder_path):

        self.vibration_dataset = vibration_dataset

        pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]
        documents = []
        for pdf_file in pdf_files:
            loader = PyMuPDFLoader(pdf_file)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device":"cuda"}, encode_kwargs={"normalize_embeddings":True})
        persist_directory = os.path.join(pdf_folder_path, "vectorstore")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        self.format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

    def __len__(self):
        return len(self.vibration_dataset)

    def __getitem__(self, idx):
        (
            description_sample, description_normal, class_name
        ) = self.vibration_dataset[idx]

        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer." 
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively," 
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        )

        user_prompt = (
            "Use the following relevant literature summaries to inform your analysis:\n"
            f"{''}\n\n"
            "You are an expert in vibration-based diagnosis of rotating machinery. "
            "Given the order frequency statistical descriptions for both the normal and current vibration signals, "
            "provide a concise diagnosis. Possible conditions: looseness, normal, unbalance, misalignment, bearing.\n\n"
            f"Normal state (x-axis): {description_normal[0]}\n"
            f"Normal state (y-axis): {description_normal[1]}\n"
            f"Current state (x-axis): {description_sample[0]}\n"
            f"Current state (y-axis): {description_sample[1]}\n"
        )

        # Retrieve RAG context based on full prompt
        rag_docs = self.retriever.get_relevant_documents(user_prompt)
        rag_context = self.format_docs(rag_docs)

        # Update user_prompt with retrieved context
        user_prompt = (
            "Use the following relevant literature summaries to inform your analysis:\n"
            f"{rag_context}\n\n"
            "You are an expert in vibration-based diagnosis of rotating machinery. "
            "Given the order frequency statistical descriptions for both the normal and current vibration signals, "
            "provide a concise diagnosis. Possible conditions: looseness, normal, unbalance, misalignment, bearing.\n\n"
            f"Normal state (x-axis): {description_normal[0]}\n"
            f"Normal state (y-axis): {description_normal[1]}\n"
            f"Current state (x-axis): {description_sample[0]}\n"
            f"Current state (y-axis): {description_sample[1]}\n"
        )

        # Generate a detailed CoT reasoning string emphasizing <VIB_EMB>
        assistant_response = f"The diagnosis result is {class_name}."

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        return {
            "prompt": prompt_only,
            "answers": assistant_response
        }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vibration_dataset = ExtendedOrderFreqDataset(data_root='/workspace/dataset')

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    llm.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)


    dataset = VibrationSFTDatasetEnglish(vibration_dataset, pdf_folder_path='./pdf')

    # 3. Configure training
    training_args = GRPOConfig(
        output_dir="output"
    )

    # 4. Initialize and train
    trainer = GRPOTrainer(
        model=llm,  
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward_format, reward_accuracy],
    )
    trainer.train()