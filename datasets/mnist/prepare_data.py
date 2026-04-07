import os
import numpy as np

from torchvision import datasets

root = "datasets/mnist"
os.makedirs(f'{root}/data', exist_ok=True)

train = datasets.MNIST(
    f"{root}/data/train", train=True, download=True, transform=None
)
print(f"Successfully prepared MNIST Dataset Train Split of Size: {len(train)}")
print(train.data.shape)
test = datasets.MNIST(
    f"{root}/data/test", train=False, download=True, transform=None
)
print(f"Successfully prepared MNIST Dataset Test Split of Size: {len(test)}")
print(test.data.shape)