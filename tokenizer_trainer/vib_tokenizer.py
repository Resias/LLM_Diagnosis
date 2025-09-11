import torch.nn as nn

class VibrationTokenizer(nn.Module):
    def __init__(self, vib_encoder, token_embed_dim, freeze_encoder=True, embedding_dim=768):
        super().__init__()
        self.vib_encoder = vib_encoder
        self.device = next(self.vib_encoder.parameters()).device
        self.dtype = next(self.vib_encoder.parameters()).dtype

        if freeze_encoder:
            for param in self.vib_encoder.parameters():
                param.requires_grad = False

        self.alignment_layer = nn.Sequential(
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

        device = next(self.vib_encoder.parameters()).device if self.vib_encoder is not None else next(self.alignment_layer.parameters()).device

        current_tensor = x.unsqueeze(0).to(device)

        class_embedding = self.vib_encoder.encode(current_tensor)

        z = self.alignment_layer(class_embedding)

        return z.detach().cpu()