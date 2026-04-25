# Federated Learning Under Adversarial Conditions

A federated learning pipeline built with Flower and PyTorch on the PathMNIST dataset. Implements dataset preprocessing, client partitioning, server coordination with PID-based anomaly detection, malicious client simulation via label-flip attacks, and results visualization.

**Authors:** Kody Rogers, Cody Smith, Nicholas Gouldin

## Getting Started

```bash
pip install -r requirements.txt
```

## Pipeline

The pipeline runs in five stages. Each stage must complete before starting the next.

### 1. Clean and normalize the dataset

Downloads PathMNIST (if not already present), removes blank images and duplicates, and normalizes pixel values to [0, 1].

```bash
python clean_data.py
```

### 2. Partition data into client datasets

Splits the cleaned data into 20 client-specific `.pt` files with labeled image directories.

```bash
python convert_clients.py
```

### 3. Run federated training

Launches the Flower server and simulated clients. Use `--attack` to enable PID-based anomaly detection mode.

```bash
python launcher.py [--attack] [--rounds ROUNDS] [--epochs EPOCHS] [--clients CLIENTS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--attack` | off | Enable attack mode with PID anomaly detection |
| `--rounds` | 10 | Number of federated communication rounds |
| `--epochs` | 5 | Local training epochs per client per round |
| `--clients` | 20 | Total number of simulated clients |

### 4. Poison and restore client data

Poison specific clients by flipping a fraction of their labels between classes 1 and 8. Restore from backup when done.

```bash
# Poison selected clients
python poison_clients.py --clients client_0.pt client_1.pt client_2.pt --flip-rate 0.25

# Restore all clients to clean state
python restore_clients.py
```

### 5. Visualize results

```bash
python plot_results.py --mode {no_attack,attack}
```

## Running the Full Attack Severity Experiment

To reproduce the severity comparison charts, run four separate simulations with fresh data each time. Use at least 20 rounds so the PID system has time to detect malicious clients.

For each severity level: restore clean data with `convert_clients.py`, poison the appropriate clients, run `launcher.py --attack`, then plot.

| Severity | Clients Poisoned | Flip Rate | Description |
|----------|-----------------|-----------|-------------|
| None | 0 / 20 | 0% | Baseline, no attack |
| Low | 3 / 20 (clients 0-2) | 25% | Subtle poisoning |
| Medium | 5 / 20 (clients 0-4) | 35% | Noticeable degradation |
| High | 8 / 20 (clients 0-7) | 60% | Strong degradation |

The PID threshold defaults to 2.79, which produced the most consistent anomaly detection results in our experiments.

After running all four simulations, generate the comparison charts:

```bash
python plot_severity_to_performance.py
```

## Project Structure

```
src/
├── clients_backup/          # Clean client data backups for restore
├── clients_data/            # Partitioned client .pt files and labeled images
├── results/
│   ├── attack/              # Training/eval CSVs and plots (attack mode)
│   └── no_attack/           # Training/eval CSVs and plots (baseline)
└── src/
    ├── clean_data.py        # Download, clean, normalize PathMNIST
    ├── convert_clients.py   # Partition dataset into client shards
    ├── launcher.py          # Flower server/client orchestrator
    ├── server_fl.py         # Server-side aggregation and PID logic
    ├── clients_fl.py        # Client-side training loop
    ├── dataset_model.py     # CNN model and dataset definitions
    ├── poison_clients.py    # Label-flip attack injection
    ├── restore_clients.py   # Restore clients from backup
    ├── plot_results.py      # Per-round training/eval visualizations
    ├── plot_severity_to_performance.py  # Cross-severity comparison charts
    └── utils_logging.py     # Logging utilities
```
