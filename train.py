import torch
from torch import nn

# INput Dim -> Hidden_dim -> mean, std -> Parametrization trick -> Decoder -> Output ime
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        #encoder
        self.img_2hdim = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        #decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hdim(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    x = torch.randn(1, 784) # MNIST, 28*28
    # vae = VAE()
    # print(vae(x))
