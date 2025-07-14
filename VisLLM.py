import torch.nn as nn
from torch.utils.data import Dataset

class VibEnbedding(nn.Module):
    def __init__(self, vib_encoder, token_embed_dim):
        self.vib_encoder = vib_encoder
        in_dim = self.vib_encoder.embedding_dim
        
        self.model = nn.Linear(
            in_features=in_dim,
            out_features=token_embed_dim
        )
    
    def __forward__(self, x_current, x_normal):
        sample_attn, normal_attn, {
                "sample_attn_scores_list": sample_scores_list,
                "normal_attn_scores_list": normal_scores_list
            } = self.vib_encoder(x_current, x_normal, get_z=True)
        current_z = self.model(sample_attn)
        normal_z = self.model(normal_attn)
        return current_z, normal_z

# Dataset
class VibrationSFTDatasetEnglish(Dataset):
    def __init__(self, vibration_dataset, tokenizer, vib_encoder, embedding_dim):

        self.vibration_dataset = vibration_dataset
        self.tokenizer = tokenizer
        self.vib_tokenizer = VibEnbedding(
            vib_encdoer = vib_encoder,
            token_embed_dim=embedding_dim
        )


    def __len__(self):
        return len(self.vibration_dataset)

    def __getitem__(self, idx):
        (
            sample_tensor, normal_tensor, row
        ) = self.vibration_dataset.__getitem__(idx, data_info=True)

        currnet_token, normal_token = self.vib_tokenizer(sample_tensor, normal_tensor)
        
        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer." 
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively," 
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        )

        # Update user_prompt with retrieved context
        user_prompt = (
            "You are an expert in vibration-based diagnosis of rotating machinery. "
            "Given the both the normal and current vibration signals, "
            "provide a concise diagnosis. Possible conditions: looseness, normal, unbalance, misalignment, bearing.\n"
            f"Normal state : <NORMAL_VIB_EMB>\n"
            f"Current state : <CURRENT_VIB_EMB>\n"
        )
        
        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        assistant_response = f"The diagnosis result is {row['class_name']}."
        
        return {
            "prompt": prompt_only,
            "answers": assistant_response,
            "normal_token" : normal_token,
            "currnet_token" : currnet_token
        }
        
if __name__ == '__main__':
    