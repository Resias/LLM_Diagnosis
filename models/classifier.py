import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_channels, in_dim, n_classes):
        super(Classifier, self).__init__()

        self.squeeze_layers = nn.Sequential(
                                nn.Flatten()
                                )
        n_features = int(in_channels*in_dim)
        self.classifier = nn.Sequential(
                                nn.Linear(n_features, n_features//2),
                                nn.ReLU(),
                                nn.Linear(n_features//2, n_features//4),
                                nn.ReLU(),
                                nn.Linear(n_features//4, n_features//8),
                                nn.ReLU(),
                                nn.Linear(n_features//8, n_classes),
                                nn.Softmax(dim=1)
                                )
    def forward(self, x):
        z = self.squeeze_layers(x)
        out = self.classifier(z)
        
        return out

class ClassifierDistance(nn.Module):
    def __init__(self, in_channels, in_dim, n_classes):
        super(ClassifierDistance, self).__init__()

        self.squeeze_layers = nn.Sequential(
                                nn.Flatten()
                                )
        n_features = int(in_channels*in_dim)
        self.classifier = nn.Sequential(
                                nn.Linear(n_features, n_features//2),
                                nn.ReLU(),
                                nn.Linear(n_features//2, n_features//4),
                                nn.ReLU(),
                                nn.Linear(n_features//4, n_features//8),
                                nn.ReLU(),
                                nn.Linear(n_features//8, n_classes),
                                nn.Softmax(dim=1)
                                )      
    def forward(self, input, ref):
        
        z_input = self.squeeze_layers(input)
        z_ref = self.squeeze_layers(ref)
        
        z_diff = z_input - z_ref                       # 방향+크기 포함

        out = self.classifier(z_diff)
        
        return out

class ClassifierUnet(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, n_classes):
        super(ClassifierUnet, self).__init__()
        n_features = embedding_dim * num_embeddings
        self.squeeze_layers = nn.Sequential(
                                nn.Flatten()
                                )
        self.classifier = nn.Sequential(
            nn.Linear(n_features, n_classes),
            # nn.Linear(n_features, n_features//2),
            # nn.Linear(n_features//2, n_classes),
            nn.Softmax(dim=1)
        )
        

    def forward(self, input, ref):
        
        z_input = self.squeeze_layers(input)
        
        z_ref = self.squeeze_layers(ref)
        
        z_diff = z_input - z_ref                       # 방향+크기 포함
        
        out = self.classifier(z_diff)
        
        return out

class SVC(nn.Module):
    def __init__(self, in_channels, in_dim, n_classes):
        super(SVC, self).__init__()

        self.squeeze_layers = nn.Sequential(
                                nn.Flatten()
                                )
        n_features = int(8*in_dim)
        self.classifier = nn.Sequential(
                                # nn.Linear(n_features, n_classes),
                                nn.Linear(n_features, n_features//2),
                                nn.Linear(n_features//2, n_classes),
                                nn.Softmax(dim=1)
                                )
    def forward(self, input, ref):

        z_input = self.squeeze_layers(input)
        z_ref = self.squeeze_layers(ref)
        
        z_diff = z_input - z_ref                       # 방향+크기 포함

        out = self.classifier(z_diff)
        
        return out

if __name__ == '__main__':
    print('StyleGAN Idea')