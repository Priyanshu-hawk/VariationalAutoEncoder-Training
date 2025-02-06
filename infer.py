import torch
from torch import nn
from model import VAE
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

# config
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
InputDim = 28*28
H_Dim = 200
Z_dim = 20
BatchSize = 32


transfrom = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

dataset = datasets.MNIST(root="dataset", train=True, transform=transfrom, download=True)
train_loader = DataLoader(dataset=dataset,
                        batch_size=BatchSize,
                        shuffle=True)

model = VAE(input_dim=InputDim, h_dim=H_Dim, z_dim=Z_dim)
model.load_state_dict(torch.load("./vae_25.pt", weights_only=True))

def infrence(digit, nums):
    # getting imgs
    imgs = []
    for x, y in dataset:
        imgs.append(x)
        if len(imgs) == 10:
            break
    
    # print(len(imgs))
    
    imgs_emded = []
    for i in imgs:
        with torch.inference_mode():
            mu, sigma = model.encode(i.view(1, 784))
        imgs_emded.append((mu, sigma))
    
    # print(len(imgs_emded))

    mu, sigma = imgs_emded[digit]
    for nums_eg in range(nums):
        epsilon = torch.randn_like(sigma)
        z = mu+sigma*epsilon
        out = model.decode(z)
        out = out.view(-1,1,28,28)
        print(out.shape)
        save_image(out, f"{nums_eg}.png")

if __name__ == "__main__":
    infrence(5, 10)