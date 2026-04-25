import torch, torch.nn as nn
from torch.utils.data import Dataset

class ClientTensorDataset(Dataset):
    """
    initialize the tensor dataset

    We are taking in a numpy dataset from pathmnist_cleaned.npz
    X = numpy image matrix derived from pathmnist_cleaned.npz
    N = sample num
    H = height
    W = width
    C = color channel count

    y = a vector of labels 0 to N with N = number of images
    """
    def __init__(self, X_np, y_np):
        # Permute the input numpy matrix from N, H, W, C to N, C, H, W
        # basically the channel is now in the 2nd dimension, not the 4th
        X = torch.from_numpy(X_np).permute(0, 3, 1, 2).float()  # NHWC->NCHW
        y = torch.from_numpy(y_np).long()

        self.X, self.y = X, y

    # simple length function
    def __len__(self): return self.X.shape[0]

    # for iteration in pytorch
    def __getitem__(self, i): return self.X[i], self.y[i]


# convolutional neural network with 9 classes
# Fairly small in size to reduce the chance of overfitting
# we made this decision because each client only receives 45 samples, 900 imgs total
class TinyCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7,128), nn.ReLU(),
            nn.Linear(128,num_classes),
        )
    def forward(self, x):
        return self.net(x)