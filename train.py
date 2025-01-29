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
NumEpochs = 10
BatchSize = 32
Lr = 3e-4

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
