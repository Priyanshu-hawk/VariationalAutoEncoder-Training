import torch
from torch import nn
import torch.nn.functional as F

# INput Dim -> Hidden_dim -> mean, std -> Parametrization trick -> Decoder -> Output ime
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        self.img_2hdim = nn.Linear(input_dim, h_dim)

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    x = torch.randn(1, 784) # MNIST, 28*28
    # vae = VAE()
    # print(vae(x))
