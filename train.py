import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
from torchvision.utils import save_image
# from torchvision.datasets import DatasetFolder
from torchvision import transforms
from tqdm import tqdm
from model import VAE

# config
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
InputDim = 28*28
H_Dim = 200
Z_dim = 20
NumEpochs = 100
BatchSize = 64
Lr = 1e-4

transfrom = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

dataset = datasets.MNIST(root="dataset", train=True, transform=transfrom, download=True)
train_loader = DataLoader(dataset=dataset,
                        batch_size=BatchSize,
                        shuffle=True)
                    
model = VAE(input_dim=InputDim, h_dim=H_Dim, z_dim=Z_dim).to(Device)
optim = torch.optim.Adam(model.parameters(), lr=Lr)

loss_fn = nn.BCELoss(reduction="sum")

# train
for epocs in range(NumEpochs):
    loop = tqdm(enumerate(train_loader))
    for i, (x,_) in loop:
        x = x.to(Device).view(x.shape[0], InputDim)
        x_recont, mu , sigma = model(x)


        recosnt_loss = loss_fn(x_recont, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        #backward
        batch_loss = recosnt_loss + kl_div
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        loop.set_postfix(loss=batch_loss.item())

torch.save(model.state_dict(), "vae_25.pt")