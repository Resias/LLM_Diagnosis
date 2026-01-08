import torch

path = "data/processed/llm_vib_validset_only_vat.pt"
data = torch.load(path)
print(type(data))
try:
    print(len(data))
except TypeError:
    print("len() 안 되는 타입:", type(data))
