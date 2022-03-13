import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_dataset = datasets.MNIST(root='./dataset/mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./dataset/mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
