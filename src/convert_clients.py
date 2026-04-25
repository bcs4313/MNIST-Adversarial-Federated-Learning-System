import numpy as np
import torch
from PIL import Image
import os
import shutil

LABEL_MAP = {
    0: "Adipose tissue",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth muscle",
    6: "Normal colon mucosa",
    7: "Cancer-associated stroma",
    8: "Colorectal adenocarcinoma epithelium"
}

# Partition clients with 5 samples per label
def partition_clients(y, num_clients=20, samples_per_label=350):
    num_classes = len(np.unique(y))
    
    indices_by_class = {}
    for class_id in range(num_classes):
        indices_for_class = np.where(y == class_id)[0]
        indices_by_class[class_id] = indices_for_class

    # Shuffle each class pool
    for label in indices_by_class:
        np.random.shuffle(indices_by_class[label])

    client_indices = []
    for client_id in range(num_clients):
        client_indices.append([])

    for label in range(num_classes):
        label_indices = indices_by_class[label]
        
        required = samples_per_label * num_clients
        selected = label_indices[:required]
        
        splits_for_clients = np.array_split(selected, num_clients)
        for client_id in range(num_clients):
            client_chunk = splits_for_clients[client_id]
            client_indices[client_id].extend(client_chunk.tolist())

    return client_indices

# -------------------------
# Save client data with label-wise numbering
# -------------------------
def save_client_data(client_id, X, y, out_dir):
    pt_path = os.path.join(out_dir, f"client_{client_id}.pt")
    torch.save((X, y), pt_path)

    img_dir = os.path.join(out_dir, f"client_{client_id}_images")
    os.makedirs(img_dir, exist_ok=True)

    # Track sample number per label
    label_counts = {}
    for label in np.unique(y):
        label_counts[label] = 0

    for i in range(len(X)):
        img = (X[i] * 255).astype(np.uint8)
        label = y[i]

        if img.shape[-1] == 3:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img.squeeze(), mode="L")

        #og_name = original_names[i] if original_names is not None else f"og{i}"
        #base_name = os.path.splitext(og_name)[0]

        sample_num = label_counts[label]
        label_counts[label] += 1

        # Save image with sample number, label, and original name

        #Debugging line
        #filename = f"sample_{sample_num}_label{label}_{base_name}.png"
        filename = f"sample_{sample_num}_label{label}.png"
        pil_img.save(os.path.join(img_dir, filename))

# -------------------------
# Main
# -------------------------
def main():
    OUT_DIR = "../clients_data"
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load("../pathmnist_cleaned.npz")
    X_train = data["train_images"]
    y_train = data["train_labels"]

    original_names = [f"Sample_{i}" for i in range(len(X_train))]

    client_indices = partition_clients(y_train, num_clients=20, samples_per_label=350)

    print("\nAssigning samples to each client...\n")
    for cid, idxs in enumerate(client_indices):
        X_client, y_client = X_train[idxs], y_train[idxs]
        orig_names_client = [original_names[i] for i in idxs]
        #save_client_data(cid, X_client, y_client, OUT_DIR, original_names=orig_names_client)
        save_client_data(cid, X_client, y_client, OUT_DIR)

    print("\nAll client datasets saved in clients_data/")

if __name__ == "__main__":
    main()
