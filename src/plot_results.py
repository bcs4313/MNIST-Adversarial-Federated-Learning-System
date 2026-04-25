import argparse, pandas as pd, matplotlib.pyplot as plt, os

def plot_curves(per_client_csv, avg_csv, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    pc = pd.read_csv(per_client_csv)
    avg = pd.read_csv(avg_csv)

    # Plotting the loss for each client over the round number
    for cid, g in pc.groupby("cid"):
        plt.figure()
        g.sort_values("round").plot(x = "round", y = "train_loss", legend=False)
        plt.title("Train Loss per Round – client" + str(cid))
        plt.savefig(os.path.join(out_dir, tag + "_train_loss_client_" + str(cid) + ".png")); plt.close()

    # Plotting the average loss among all clients per round
    plt.figure()
    avg.plot(x = "round", y = "avg_train_loss", legend=False)
    plt.title("Average Train Loss per Round")
    plt.savefig(os.path.join(out_dir, tag + "_avg_train_loss.png")); plt.close()

    # Plotting the acc for each client over the round number
    # Join eval per-client
    ev = pd.read_csv(per_client_csv.replace("per_client_train.csv","per_client_eval.csv"))
    for cid, g in ev.groupby("cid"):
        plt.figure()
        g.sort_values("round").plot(x = "round", y = "val_acc", legend=False)
        plt.title("Val Acc per Round – client " + str(cid))
        plt.savefig(os.path.join(out_dir, tag + "_val_acc_client_" + str(cid) + ".png")); plt.close()

    # Plotting the average acc among all clients per round
    plt.figure()
    avg.plot(x = "round", y = "avg_val_acc", legend=False)
    plt.title("Average Val Acc per Round")
    plt.savefig(os.path.join(out_dir, tag + "_avg_val_acc.png")); plt.close()

def plot_pid(pid_csv, out_dir, tag):
    df = pd.read_csv(pid_csv)
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣  Rejected-client count per round
    rejected_per_round = df.groupby("round_num")["rejected"].sum()
    plt.figure()
    rejected_per_round.plot(marker="o")
    plt.title("Rejected Clients per Round (PID)")
    plt.xlabel("Round")
    plt.ylabel("Rejected Clients")
    plt.savefig(os.path.join(out_dir, f"{tag}_pid_rejections_per_round.png"))
    plt.close()

    # 2️⃣  Optional (nice to show PID behavior separation)
    plt.figure()
    for cid, g in df.groupby("cid"):
        short_label = str(cid).replace("client_", "")
        plt.plot(g["round_num"], g["anomaly_score"], label=short_label, alpha=0.7)
    plt.axhline(y=df["anomaly_score"].mean(), color="r", linestyle="--", label="mean anomaly score")
    plt.title("PID Score per Client per Round")
    plt.xlabel("Round")
    plt.ylabel("PID Score (anomaly_score)")
    plt.legend(
        title="CID",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize="small",
        ncol=2,  # two columns in legend
        columnspacing=0.8,
        handlelength=1.0
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_pid_scores_per_client.png"))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_attack","attack"], required=True)
    args = ap.parse_args()
    root = f"../results/{args.mode}"
    plot_curves(os.path.join(root,"per_client_train.csv"),
                os.path.join(root,"round_averages.csv"),
                root, tag=args.mode)
    plot_pid(os.path.join(root,"pid_history.csv"), root, tag=args.mode)

if __name__ == "__main__":
    main()