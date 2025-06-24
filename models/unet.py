import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self,
                in_channels,
                num_residual_hiddens):
        super(Residual, self).__init__()
        
        num_residual_hiddens = int(in_channels*num_residual_hiddens)
        
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=in_channels,
                    out_channels=num_residual_hiddens,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=num_residual_hiddens,
                    out_channels=in_channels,
                    kernel_size=1, stride=1, bias=False
                    )
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class ResidualStack(nn.Module):
    def __init__(self,
                in_channels,
                num_residual_layers,
                num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([
            Residual(in_channels, num_residual_hiddens) for _ in range(self.num_residual_layers)
        ])
    
    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self,
                in_length,
                in_channels,
                embedding_dim,
                num_embeddings,
                num_residual_layers,
                num_residual_hiddens):
        super(Encoder, self).__init__()
        """
        in_channels x in_length -> embedding_dim x num_embeddings 으로 압축
        """
        
        current_ch = in_channels
        current_length = in_length        
        
        ch_expand = 0
        while True:
            if current_ch == embedding_dim:
                break
            current_ch *= 2
            ch_expand += 1
        len_squeeze = 0
        while True:
            if current_length == num_embeddings:
                break
            current_length //= 2
            len_squeeze += 1
        
        
        # encoding strategy
        schedule = bresenham_style_decrement(ch_expand, len_squeeze)
        
        conv_layers = nn.ModuleList()
        current_ch = in_channels
        current_length = in_length
        for strtegy in schedule:       
            
            # 채널수만 늘리기
            if strtegy =='ch':
                next_ch = int(current_ch * 2)
                conv_layers.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=current_ch,
                        out_channels=next_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                    nn.BatchNorm1d(next_ch),
                    ResidualStack(
                        in_channels=next_ch,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens
                    ),
                ))
                current_ch = next_ch
            
            # 길이만 줄이기
            elif strtegy == 'len':
                next_length = current_length // 2
                conv_layers.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=current_ch,
                        out_channels=current_ch,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm1d(current_ch),
                    ResidualStack(
                        in_channels=current_ch,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens
                    ),
                ))
                current_length = next_length
            
            # 채널수 늘리고 길이 줄이기
            elif strtegy == 'both':
                next_ch = int(current_ch * 2)
                next_length = current_length // 2
                conv_layers.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=current_ch,
                        out_channels=next_ch,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm1d(next_ch),
                    ResidualStack(
                        in_channels=next_ch,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens
                    ),
                ))
                current_ch = next_ch
                current_length = next_length
            
            
        
        self.num_conv_layers = len(conv_layers)
        self.conv_layers = conv_layers
        self.num_embeddings = current_length
        self.embedding_dim = current_ch
        self.schedule = schedule
    
    def forward(self, x, return_skip=False):
        skips = []
        
        for idx, block in enumerate(self.conv_layers):
            x = block(x)
            if return_skip:
                skips.append(x)
        
        return (x, skips) if return_skip else x
    
class Decoder(nn.Module):
    def __init__(self,
                    out_length,
                    out_channels,
                    embedding_dim,
                    num_embeddings,
                    num_residual_layers,
                    num_residual_hiddens,
                    schedule):
        super(Decoder, self).__init__()

        self.schedule = schedule[::-1]  # 역순으로 적용
        self.layers = nn.ModuleList()

        current_ch = embedding_dim
        current_length = num_embeddings

        for idx, step in enumerate(self.schedule):
            if step == 'ch':
                next_ch = current_ch // 2
                current_ch *= 2

                block = nn.Sequential(
                    nn.Conv1d(current_ch, next_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(next_ch),
                    ResidualStack(
                        in_channels=next_ch,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens
                    )
                )
                self.layers.append(block)
                current_ch = next_ch

            elif step == 'len':
                next_length = current_length * 2
                current_ch *= 2

                block = nn.Sequential(
                    nn.ConvTranspose1d(current_ch, current_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(current_ch),
                    ResidualStack(
                        in_channels=current_ch,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens
                    )
                )
                self.layers.append(block)
                current_length = next_length

            elif step == 'both':
                next_ch = current_ch // 2
                next_length = current_length * 2
                current_ch *= 2

                block = nn.Sequential(
                    nn.ConvTranspose1d(current_ch, next_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(next_ch),
                    ResidualStack(
                        in_channels=next_ch,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens
                    )
                )
                self.layers.append(block)
                current_ch = next_ch
                current_length = next_length

        self.final_ch = current_ch
        self.final_length = current_length
        self.out_channels = out_channels

        self.final_conv = nn.Conv1d(current_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skips=None):
        if skips is None:
            skips = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            skip_feat = skips[-(i + 1)]  # 역순 대응
            
            if skip_feat is not None:
                x = torch.cat([x, skip_feat], dim=1)
            
            x = layer(x)

        return self.final_conv(x)


# --------------------
# VAE Core
# --------------------
class VAE(nn.Module):
    def __init__(self, z_channels):
        super(VAE, self).__init__()

        # Bottleneck 분리: mu, logvar 따로
        self.mu_head = nn.Conv1d(z_channels, z_channels, kernel_size=1)
        self.logvar_head = nn.Conv1d(z_channels, z_channels, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, encoded_feature):
        mu = self.mu_head(encoded_feature)
        logvar = self.logvar_head(encoded_feature)
        z = self.reparameterize(mu, logvar)

        # KL Divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return z, kl

# --------------------
# Full Unet-VAE
# --------------------

class UnetVAE(nn.Module):
    def __init__(self, 
                in_length, 
                in_channels, 
                num_residual_layers, 
                num_residual_hiddens, 
                num_embeddings, 
                embedding_dim, 
                not_vae=False):
        super().__init__()
        
        
        self.encoder = Encoder(
                in_length = in_length,
                in_channels = in_channels,
                embedding_dim = embedding_dim,
                num_embeddings  = num_embeddings,
                num_residual_layers = num_residual_layers,
                num_residual_hiddens = num_residual_hiddens
            )
        
        embedding_dim = self.encoder.embedding_dim
        
        self.vae = VAE(embedding_dim)

        self.decoder = Decoder(
                out_length = in_length,
                out_channels = in_channels,
                embedding_dim = embedding_dim,
                num_embeddings = num_embeddings,
                num_residual_layers = num_residual_layers,
                num_residual_hiddens = num_residual_hiddens,
                schedule= self.encoder.schedule
            )
        

    def forward(self, x):
        z, skips = self.encoder(x, return_skip=True)

        z, kl = self.vae(z)
        
        x_recon = self.decoder(z, skips=skips)
        
        return x_recon, kl

    def encode(self, x, return_before_vae=True):
        z, _ = self.encoder(x, return_skip=True)
        
        if return_before_vae:
            return z
        else:
            z, _ = self.vae(z)
            return z

    def tensor_transform_checker(self, x):
        print(f'input shape: {x.shape}')
        z, skips = self.encoder(x, return_skip=True)
        print(f'after encoder: {z.shape}')
        z, kl = self.vae(z)
        print(f'after vae: {z.shape}')
        x_recon = self.decoder(z, skips=skips)
        print(f'output: {x_recon.shape}')
        return x_recon, kl


def bresenham_style_decrement(ch_expand, len_squeeze):
    a, b = ch_expand, len_squeeze
    schedule = []

    x, y = 0, 0
    while a > 0 or b > 0:
        if a == 0:
            schedule.append('len')
            b -= 1
        elif b == 0:
            schedule.append('ch')
            a -= 1
        elif a * y < b * x:
            schedule.append('len')
            b -= 1
            y += 1
        elif a * y > b * x:
            schedule.append('ch')
            a -= 1
            x += 1
        else:
            schedule.append('both')
            a -= 1
            b -= 1
            x += 1
            y += 1
    return schedule


if __name__ == "__main__":
    in_length = 256
    in_channels = 2
    num_hiddens = 0 # dummy
    z_dim = 64
    num_residual_layers = 4
    num_residual_hiddens = 2
    num_embeddings = 128
    embedding_dim = 64
    not_vae = False
    
    model = UnetVAE(in_length, in_channels, z_dim, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, not_vae)
    x = torch.randn(1, in_channels, in_length)
    
    model.tensor_transform_checker(x)