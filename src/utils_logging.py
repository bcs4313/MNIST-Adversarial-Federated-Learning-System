import csv, os, statistics as stats

"""
CSV logger for federated rounds
Creates and maintains three files in the output directory 'out_dir'
- per_client_train.csv : per-client train metrics per round
- per_client_eval.csv : per-client eval metrics per round
- round_averages.csv : round-level averages and counts
- pid_history.csv: pid history for each client per round, including if they were rejected
"""
class RoundLogger:

    """
    ensure csv headers exist, and prepare the output directories by
    writing their initial columns
    """
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.fit_csv = os.path.join(out_dir, "per_client_train.csv")
        self.eval_csv = os.path.join(out_dir, "per_client_eval.csv")
        self.avg_csv = os.path.join(out_dir, "round_averages.csv")
        self.pid_csv = os.path.join(out_dir, "pid_history.csv")
        # headers (create if missing)
        if not os.path.exists(self.fit_csv):
            with open(self.fit_csv, "w", newline="") as f:
                csv.writer(f).writerow(["round","cid","train_loss","train_acc","num_examples"])
        if not os.path.exists(self.eval_csv):
            with open(self.eval_csv, "w", newline="") as f:
                csv.writer(f).writerow(["round","cid","val_loss","val_acc","num_examples"])
        if not os.path.exists(self.avg_csv):
            with open(self.avg_csv, "w", newline="") as f:
                csv.writer(f).writerow(["round","avg_train_loss","avg_train_acc","avg_val_loss","avg_val_acc","clients_used","clients_failed"])
        if not os.path.exists(self.avg_csv):
            with open(self.pid_csv, "w", newline="") as f:
                csv.writer(f).writerow(["round", "cid", "dist", "anomaly_score", "rejected"])

        # write the starter row of the pid csv
        with open(self.pid_csv, "a", newline="") as file:
            w = csv.writer(file)
            w.writerow(["round_num", "cid", "dist", "anomaly_score", "rejected"])

    # append per client training rows for a given round
    def log_fit_rows(self, round_num, rows):
        with open(self.fit_csv, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows: w.writerow([round_num]+r)

    # append per client evaluation rows for a given round
    def log_eval_rows(self, round_num, rows):
        with open(self.eval_csv, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows: w.writerow([round_num]+r)

    # append round-level averages and counts across ALL clients
    def log_avgs(self, round_num, fit_rows, eval_rows, failures):
        def avg(vals): return sum(vals)/len(vals) if vals else float("nan")
        avg_train_loss = avg([r[1] for r in fit_rows])  # r = [cid, train_loss, train_acc, n]
        avg_train_acc  = avg([r[2] for r in fit_rows])
        avg_val_loss   = avg([r[1] for r in eval_rows]) # r = [cid, val_loss, val_acc, n]
        avg_val_acc    = avg([r[2] for r in eval_rows])
        with open(self.avg_csv, "a", newline="") as f:
            csv.writer(f).writerow([round_num, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, len(fit_rows), len(failures)])

    # PID logging: batched in rows
    def log_pid_rows(self, round_num, rows):
        """
        rows: list of ["cid", "dist", "anomaly_score", "rejected"]
        """
        with open(self.pid_csv, "a", newline="") as file:
            w = csv.writer(file)
            for cid, dist, anomaly_score, rejected in rows:
                w.writerow([round_num, cid, float(dist), float(anomaly_score), int(bool(rejected))])