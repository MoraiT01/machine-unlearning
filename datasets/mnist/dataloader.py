import numpy as np
from torchvision import datasets
from typing import Tuple

root = "datasets/mnist"

# Initialize datasets without converting to NumPy yet
# torchvision.datasets.MNIST still loads the .pt files into memory,
# but we avoid creating the additional large NumPy copies.
train_data = datasets.MNIST(root, train=True, download=True)
test_data = datasets.MNIST(root, train=False, download=True)

def load(indices, category='train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches only the specific samples requested and returns them as NumPy arrays.
    """
    if category not in ['train', 'test']:
        raise ValueError("Category must be 'train' or 'test'")
    dataset = train_data if category == 'train' else test_data
    
    images = []
    labels = []
    
    for idx in indices:
        # dataset[idx] returns a tuple: (PIL Image, Label)
        img, label = dataset[idx]
        
        # Convert PIL image to NumPy array and normalize if needed
        images.append(np.array(img))
        labels.append(label)
    
    # Stack the list of arrays into a single batch (Batch Size, Height, Width)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
