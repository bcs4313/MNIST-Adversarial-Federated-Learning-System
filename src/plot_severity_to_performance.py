import argparse, pandas as pd, matplotlib.pyplot as plt, os
from pathlib import Path

# this is a special plotting file that takes in
# round_averages.csv from the following directories:
# "no_attack", "attack_low", "attack_med", "attack_high" from results
# and plots a bar graph of the mean loss and acc from these severities
def plot_attack_severity(results_root, severities, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)   # <-- ensure folder exists

    data = []
    for tag in severities:
        csv_path = Path(results_root) / tag / "round_averages.csv"
        if not csv_path.exists():
            print(f"Skipping {csv_path} (missing)")
            continue
        df = pd.read_csv(csv_path)
        final = df.iloc[-1]
        data.append({
            "severity": tag,
            "final_train_loss": final["avg_train_loss"],
            "final_val_acc": final["avg_val_acc"],
        })

    if not data:
        print("No data found; nothing to plot.")
        return

    df = pd.DataFrame(data)

    plt.figure()
    plt.bar(df["severity"], df["final_val_acc"])
    plt.title("Final Validation Accuracy vs Attack Severity")
    plt.ylabel("Final Avg Val Accuracy")
    plt.xlabel("Attack Severity")
    plt.ylim(0, 1)
    plt.savefig(out_dir / "accuracy_vs_attack_severity.png")
    plt.close()

    plt.figure()
    plt.bar(df["severity"], df["final_train_loss"])
    plt.title("Final Train Loss vs Attack Severity")
    plt.ylabel("Final Avg Train Loss")
    plt.xlabel("Attack Severity")
    plt.savefig(out_dir / "loss_vs_attack_severity.png")
    plt.close()

if __name__ == "__main__":
    # if you run from src/, these resolve to the project root correctly
    base = Path(__file__).resolve().parents[1]
    plot_attack_severity(
        results_root=base / "results",
        severities=["no_attack", "attack_low", "attack_med", "attack_high"],
        out_dir=base / "results" / "summary",
    )
