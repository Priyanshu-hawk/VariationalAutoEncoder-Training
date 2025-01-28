import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
from torchvision.utils import save_image
# from torchvision.datasets import DatasetFolder
from torchvision import transforms

from model import VAE

# config
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
InputDim = 28*28
H_Dim = 20
z_dim = 20 