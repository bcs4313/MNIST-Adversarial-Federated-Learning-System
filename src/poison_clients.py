import os
import shutil
import torch
import argparse

# Directories relative to this script
BASE_DIR = os.path.dirname(__file__)
CLIENTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "clients_data"))
BACKUP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "clients_backup"))
os.makedirs(BACKUP_DIR, exist_ok=True)

def poison_targeted(client_path, flip_pairs=[(1, 8)], flip_rate=1.0):
    """Flip selected label pairs with optional flip rate (symmetric)."""
    X, y = torch.load(client_path, weights_only=False)
    y = torch.tensor(y) if not torch.is_tensor(y) else y.clone()
    y_orig = y.clone()

    total_flipped = 0
    for a, b in flip_pairs:
        mask_a = (y_orig == a).nonzero(as_tuple=True)[0]
        mask_b = (y_orig == b).nonzero(as_tuple=True)[0]

        if flip_rate < 1.0:
            n_a = int(len(mask_a) * flip_rate)
            n_b = int(len(mask_b) * flip_rate)
            mask_a = mask_a[torch.randperm(len(mask_a))[:n_a]]
            mask_b = mask_b[torch.randperm(len(mask_b))[:n_b]]

        y[mask_a] = b
        y[mask_b] = a
        total_flipped += len(mask_a) + len(mask_b)

    print(f"[INFO] Flipped {total_flipped} samples in {os.path.basename(client_path)}")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Poison federated client data")
    parser.add_argument("--clients", nargs="+", required=True, help="Client filenames (e.g., client_0.pt client_1.pt)")
    parser.add_argument("--flip-rate", type=float, default=1.0, help="Fraction of samples to flip (0.0–1.0)")
    args = parser.parse_args()

    for client_name in args.clients:
        # Allow user to just say "client_0.pt" or "client_0"
        if not client_name.endswith(".pt"):
            client_name += ".pt"

        src_path = os.path.join(CLIENTS_DIR, client_name)
        if not os.path.exists(src_path):
            print(f"[ERROR] Could not find {src_path}")
            continue

        backup_path = os.path.join(BACKUP_DIR, client_name)
        shutil.copy2(src_path, backup_path)
        print(f"[INFO] Backed up {client_name} → {BACKUP_DIR}")

        X_poisoned, y_poisoned = poison_targeted(src_path, flip_pairs=[(1, 8)], flip_rate=args.flip_rate)

        # Overwrite the original client file in clients_data
        torch.save((X_poisoned, y_poisoned), src_path)
        print(f"[INFO] Overwrote {client_name} with poisoned data in {CLIENTS_DIR}")


if __name__ == "__main__":
    main()
