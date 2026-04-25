# so many imports...
import argparse, os
import numpy as np
import torch
import flwr as fl
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_model import ClientTensorDataset, TinyCNN

"""
Federated Flower Client
- Client that trains a small CNN on a localized dataset.
- Has support for label flip attacks to simulate malicious clients
- and uses training and validation locally
"""

import sys
try:
    sys.stdout.reconfigure(line_buffering=True)  # Py 3.7+
except Exception:
    pass

"""
load the MNIST dataset from a .pt file on disk
@param: pt_path : str - path to the pytorch file with the client's samples
found in clients_data and client_X.pt with X being the client number
@return numpy matrix of images and labels
"""
def load_np(pt_path):
    print(f"[client] loading: {pt_path}")

    # PyTorch 2.6: default weights_only=True blocks pickled numpy
    X_np, y_np = torch.load(pt_path, map_location="cpu", weights_only=False)

    # If someone saved tensors instead of numpy, convert back:
    if hasattr(X_np, "numpy"):
        X_np = X_np.numpy()

    if hasattr(y_np, "numpy"):
        y_np = y_np.numpy()
    return X_np, y_np

"""
Simulates a label flip attack from a malicious client

@param y : np.ndarray - original vector of labels
@param src : int - class to flip
@param dst : int - class to flip into
@param frac: float - labels to flip, by (range 0.0-1.0)
@param seed: rng seed
@return np.ndarray - new label vector that applied the flips
"""
def apply_label_flip(y, src=8, dst=1, frac=1.0, seed=0):
    np.random.seed(seed)
    y = y.copy()
    idx = np.where(y == src)[0]
    k = int(len(idx) * frac)
    if k > 0:
        flip = np.random.choice(idx, k, replace=False)
        y[flip] = dst
    return y

"""
Create train and validation dataloader objects from our numpy dataset
- Split is 80/20, with a default seed for reproducibility
@param X_np, y_np : np.ndarray - Input images and labels from MNIST dataset
@param batch : int - batch size
@param seed : int - rng seed
@return (train loader, val loader) : Tuple[DataLoader, DataLoader]
"""
def make_loaders(X_np, y_np, batch=64, seed=42):
    CTDataset = ClientTensorDataset(X_np, y_np)
    n_tr = int(0.8 * len(CTDataset))
    n_val = len(CTDataset) - n_tr
    generator_torch = torch.Generator().manual_seed(seed)
    tr, va = random_split(CTDataset, [n_tr, n_val], generator=generator_torch)
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(va, batch_size=batch)

"""
run epoch on train split
@param model : nn.Module - Our TinyCNN model we defined inside dataset_model.py
@param loader: DataLoader - just the data loader
@param crit : optimizer - our cross entropy loss function
@param opt: optimizer - Using Adam with a small-ish learning ratge
@param dev : device - Specify 'cpu' or 'cuda' based on what system you have

@return dict with keys {"train_loss": float, "train_acc": float}
"""
def train_one_epoch(model, loader, crit, opt, dev):
    model.train()
    trained_loss = 0
    total_samples = 0
    correct = 0

    for xb,yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb); loss.backward(); opt.step()
        trained_loss += loss.item() * yb.size(0);
        total_samples += yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()

    return {"train_loss": trained_loss / total_samples, "train_acc": correct / total_samples}

"""
Get loss and acc on the validation set
@param model: nn.Module
@param loader: DataLoader
@param crit: loss function
@param dev: device - cpu or cuda again

@return dict with keys {"train_loss": float, "train_acc": float}
"""
def evaluate(model, loader, crit, dev):
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_count = 0

    with (torch.no_grad()):
        for xb,yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            out = model(xb)
            loss = crit(out, yb)

            total_loss += loss.item()*yb.size(0)
            total_samples += yb.size(0)
            correct_count += (out.argmax(1) == yb).sum().item()

    return {"val_loss": total_loss/total_samples, "val_acc": correct_count/total_samples}

"""tiny federated client with optional label flipping
- loads local dataset from disk
- trains for X epochs and outputs key metrics
"""
class Client(fl.client.NumPyClient):

    """Constructor for class
    @param cid: int - client id, maps to file names in the MNIST directory
    @param pt_path: str path to the client (cid injected in the middle)
    @param epochs : int
    @param malicious : bool - does the client do label flip attacks at all?
    @param flip_src,flip_dst,flip_frac - label flip settings
    @param device: str - specify if you are working with cpu or cuda here
    """
    def __init__(self, cid, pt_path, epochs=1, malicious=False, flip_src=8, flip_dst=1, flip_frac=0.0, device="cpu"):
        print("loading: " + pt_path)
        self.cid = str(cid)
        X_np, y_np = load_np(pt_path)
        self.data_name = os.path.splitext(os.path.basename(pt_path))[0]
        if malicious and flip_frac>0:
            y_np = apply_label_flip(y_np, src=flip_src, dst=flip_dst, frac=flip_frac, seed=cid)
        self.train_loader, self.val_loader = make_loaders(X_np, y_np)
        self.device = torch.device(device)
        self.model = TinyCNN().to(self.device)
        self.crit = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = epochs

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = self.model.state_dict()
        for (k,_), arr in zip(sd.items(), params):
            sd[k] = torch.from_numpy(arr)
        self.model.load_state_dict(sd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics={}
        for _ in range(self.epochs):
            metrics = train_one_epoch(self.model, self.train_loader, self.crit, self.opt, self.device)
            metrics["data_name"] = self.data_name

        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        m = evaluate(self.model, self.val_loader, self.crit, self.device)
        m["data_name"] = self.data_name

        return float(m["val_loss"]), len(self.val_loader.dataset), m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--data_dir", type=str, default="clients_data")
    ap.add_argument("--server", type=str, default="127.0.0.1:8080")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--malicious", action="store_true")
    ap.add_argument("--flip_src", type=int, default=8)
    ap.add_argument("--flip_dst", type=int, default=1)
    ap.add_argument("--flip_frac", type=float, default=0.0)  # 0.0 for baseline
    args = ap.parse_args()

    pt = os.path.join(args.data_dir, f"client_{args.cid}.pt")
    client = Client(args.cid, pt, epochs=args.epochs,
                    malicious=False,  # clients are assumed to be non-malicious
                    flip_src=args.flip_src, flip_dst=args.flip_dst, flip_frac=args.flip_frac)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
