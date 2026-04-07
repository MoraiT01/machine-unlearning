import numpy as np
import os

roots = "datasets/purchase/data"

train_data = np.load(os.path.join(roots, 'purchase2_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(roots, 'purchase2_test.npy'), allow_pickle=True)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]

X_train = train_data['X'].astype(np.float32)
X_test = test_data['X'].astype(np.float32)
y_train = train_data['y'].astype(np.int64)
y_test = test_data['y'].astype(np.int64)

def load(indices, category='train'):

    if category not in ['train', 'test']:
        raise ValueError("Category must be 'train' or 'test'")
    dataset = train_data if category == 'train' else test_data
    
    inputs = []
    outputs = []
    
    for idx in indices:
        # dataset[idx] returns a tuple: (PIL Image, Label)
        img, label = dataset["X"][idx], dataset["y"][idx]
        
        # Convert PIL image to NumPy array and normalize if needed
        inputs.append(np.array(img))
        outputs.append(label)
    
    # Stack the list of arrays into a single batch (Batch Size, Height, Width)
    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.int64)