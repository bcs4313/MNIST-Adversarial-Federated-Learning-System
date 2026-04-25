### Authors
Kody Rogers
Cody Smith
Nicholas Gouldin

### INTRODUCTION
This project provides a pipeline for downloading, cleaning, and partitioning the PathMNIST dataset
into client-specific datasets for experiments such as federated learning or distributed training.
This repository implements a federated learning pipeline using Flower (FLWR) and PyTorch on the PathMNIST dataset.
It includes dataset preprocessing, client partitioning, server coordination, malicious client simulation 
(label-flip attacks), and results visualization.


### Project Structure
src/
  ├── clients_backup/
  │   └── client_0.pt
  │   └── ...
  ├── clients_data/
  │   └── client_0_images/
  │       ├── sample_0_label0.png
  │       ├── sample_0_label1.png 
  │       ├── ...
  │       └── sample_0_label8.png
  │   └── client_1_images/...
  │   └── client_0.pt
  │   └── ...
  │   └── client_19.pt
  ├── results/
  │   └── attack/
  │       ├── attack_avg_train_loss.png
  │       ├── attack_avg_val_acc.png
  │       ├── attack_train_loss_client_0.png
  │       ├── ...
  │       ├── attack_train_loss_client_19.png
  │       ├── attack_val_acc_client_0.png
  │       ├── ...
  │       ├── attack_val_acc_client_19.png
  │       ├── per_client_eval.csv
  │       ├── per_client_train.csv
  │       └── round_averages.csv
  │   └── no_attack/
  │       ├── no_attack_avg_train_loss.png
  │       ├── no_attack_avg_val_acc.png
  │       ├── no_attack_train_loss_client_0.png
  │       ├── ...
  │       ├── no_attack_train_loss_client_19.png
  │       ├── no_attack_val_acc_client_0.png
  │       ├── ...
  │       ├── no_attack_val_acc_client_19.png
  │       ├── per_client_eval.csv
  │       ├── per_client_train.csv
  │       └── round_averages.csv
  └── src/
      ├── clean_data.py
      ├── clients_fl.py
      ├── convert_clients.py
      ├── dataset_model.py
      ├── launcher.py
      ├── plot_results.py
      ├── plot_severity_to_performance.py
      ├── poison_clients.py
      ├── restore_clients.py
      ├── server_fl.py
      └── utils_logging.py

### To Run this Code
First install the dependencies using 
pip install -r requirements.txt

# Second Step
run clean_data.py
usage: python clean_data.py
This will take a while as it needs to download the data if it isn't provided
This will also normalize the data and clean it by removing duplicates and blanks

# Third Step
run convert_clients.py
usage: python clean_data.py
This will put the data into the client_data folder which will partition it and label the data

# Fourth Step
run launcher.py
usage: launcher.py [-h] [--attack] [--rounds ROUNDS] [--epochs EPOCHS] [--clients CLIENTS]
Notes: poison_clients.py is required to create malicious clients. The launcher does not have a built in option to do this.
options:
  -h, --help            show this help message and exit
  --attack              Enable attack mode
                        Label-flip fraction
  --rounds ROUNDS       Number of federated rounds
  --epochs EPOCHS       Local epochs per client per round
  --clients CLIENTS     Total number of simulated clients

# Fifth Step
run poison_clients.py and restore_poison.py
usage: poison_clients.py [-h] --clients CLIENTS [CLIENTS ...] [--flip-rate FLIP_RATE]

Poison federated client data
options:
  -h, --help            show this help message and exit
  --clients CLIENTS [CLIENTS ...]
                        Client filenames (e.g., client_0.pt client_1.pt)
  --flip-rate FLIP_RATE
                        Fraction of samples to flip (0.0–1.0)

This will flip the data for labels 1 and 8 saved. at the percentage listed below
to revert the changes use restore_poison.py
usage: python restore_poison.py

# Last STEP
To visualize results use plot_results.py
usage: plot_results.py [-h] --mode {no_attack,attack}

options:
  -h, --help            show this help message and exit
  --mode {no_attack,attack}

## Plotting Attack Severity vs Loss/Accuracy
Run 4 separate simulations of the previous steps, each simulation running
poison_clients.py at a specific --flip-rate value:

Please run AT LEAST 20 rounds to ensure that the pid system has time to detect the malicious clients

It is assumed that you delete data in clients_data and use convert_clients.py to produce fresh
data to poison.

### Levels of flipped labels for each attack type:

#### no_attack
0.00% (baseline)

#### attack_low

--flip-rate 0.25

3 clients out of 20 poisoned (clients 0-2)

(20% of class labels flipped) — subtle poisoning

#### attack_med

--flip-rate 0.35

5 clients out of 20 poisoned (clients 0-4)

(35% of class labels flipped) — noticeable degradation

#### attack_high

--flip-rate 0.60

8 clients out of 20 poisoned (clients 0-7)

(60% of class labels flipped) — strong degradation

Our pid threshold defaults to 2.79, which seems to produce the most consistent anomaly detection results

#### Finally: Run plot_severity_to_performance.py. 

A chart giving the differences in accuracy and loss for each attack type will be saved

