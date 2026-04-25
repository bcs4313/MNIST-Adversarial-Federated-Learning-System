import argparse, flwr as fl

import numpy
from flwr.common import parameters_to_ndarrays

from utils_logging import RoundLogger

import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

"""
Aggregation class that additionally:
- stashes train metrics after aggregate_fit
- reads per-client evaluation metrics after aggregate_evaluate
- pushes both metrics into utils_logging, which are then written to the results folder
- called: per_client_eval.csvm per_client_train.csv, round_averages.csv
"""
class LoggedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, logger, kp=1.0, ki=0.1, kd=0.05, pid_threshold=0.5, total_rounds=20, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.rnd = 0  # start at 0; we'll increment at the start of each round
        self.cid_map = {}  # client_id to client_name mapping

        # list of client names that had a PID from the last round that was too high
        self.clients_to_reject = []

        # Parameters below are associated with PID in Project 3:::
        # we are using these variables to match the formula in the writeup
        # kp = coefficient for the euclidean distance of weights,
        # this coefficient measures the instantaneous deviation of the client
        # model from the server model
        # ki = coefficient for the accumulative amount of deviation a client has
        # from the server model
        # kd = coefficient for how fast the model is changing relative to the
        # server model
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.total_rounds = int(total_rounds)
        # established threshold for a client, essentially the anomaly measure
        # if a client is higher than this threshold, they are considered malicious
        # and are DISCARDED from further training and communication
        self.pid_threshold = float(pid_threshold)

        # use this standard as a baseline for evaluating anomalous models
        # --kp 1.0 --ki 0.1 --kd 0.05 --pid_threshold 0.5
        self._pid_hist = {}  # cid -> {"sum": float, "prev": float}

        # PID-based malicious labeling (server-side; no client flag required)
        self.pid_patience = int(kwargs.pop("pid_patience", 2))  # rounds in a row to label as malicious
        self._pid_reject_streak = {}  # cid -> consecutive reject count
        self._malicious = set()  # permanently labeled malicious cids

    # REMOVE on_round_begin (Flower might not call it)

    def aggregate_fit(self, server_round, results, failures):
        # <-- increment round here (called once per round)
        self.rnd += 1

        # Collect per-client TRAIN metrics
        fit_rows = []
        cids = []
        params_list = []
        for client, fitres in results:
            m = fitres.metrics or {}
            cid = m.get("data_name")
            cids.append(cid)

            if cid in self.clients_to_reject:
                print("[server] Rejected data from cid " + str(cid) + " due to malicious label")
                continue

            # collect numpy params for centroid/PID
            try:
                params_list.append(parameters_to_ndarrays(fitres.parameters))
            except Exception:
                params_list.append(None)  # keep shape consistent; we'll guard below

            tr_loss = float(m.get("train_loss", float("nan")))
            tr_acc  = float(m.get("train_acc",  float("nan")))
            n = fitres.num_examples
            fit_rows.append([cid, tr_loss, tr_acc, n])

        # --- PID filter (minimal insertion) ---
        filtered_results = results  # default: no filtering
        try:
            if params_list and all(p is not None for p in params_list):
                centroid = self._centroid(params_list)
                # compute PID scores and keep mask
                keep_mask = []
                excluded = []  # excluded client list
                pid_rows = []  # for logging in the .csv
                for cid, p in zip(cids, params_list):
                    dist = self._model_distance(p, centroid)
                    u = self._compute_pid(cid, dist)

                    print("[server] anomalyRound" + str(server_round) + " -> cid = " + str(cid) + " distance to centroid: " + str(dist) + " pid total: " + str(u))

                    keep = (u <= self.pid_threshold)  # exclude if PID < threshold (per write-up)
                    keep_mask.append(keep)
                    # I wish this was just !keep like normal languages lol
                    pid_rows.append([cid, dist, u, not keep])
                    if not keep: excluded.append(cid)

                # log our pid data
                self.logger.log_pid_rows(self.rnd, pid_rows)
                kept_count = sum(keep_mask)
                if kept_count == 0:
                    print(f"[server] PID would exclude all {len(results)} clients; falling back to prev.", flush=True)
                else:
                    filtered_results = [r for r, k in zip(results, keep_mask) if k]
                    print(
                        f"[server] Round {self.rnd} PID filter: kept={kept_count} "
                        f"excluded={len(results)-kept_count} | kp={self.kp} ki={self.ki} kd={self.kd} thr={self.pid_threshold}",
                        flush=True
                    )
                    if excluded:
                        print(f"[server]  excluded: {excluded}", flush=True)
                        self.clients_to_reject = excluded

        except Exception as e:
            print(f"[server] PID computation failed: {e}; proceeding without filtering.", flush=True)

        self._fit_rows_cache = fit_rows
        return super().aggregate_fit(server_round, filtered_results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        print("Aggregate Evaluation Phase: Checking Clients for Anomalies via PID.")
        # Collect per-client EVAL metrics
        eval_rows = []
        for client, evres in results:
            m = evres.metrics or {}
            val_loss = float(evres.loss)
            val_acc  = float(m.get("val_acc", float("nan")))
            n = evres.num_examples

            cid = m.get("data_name", client.cid)
            eval_rows.append([cid, val_loss, val_acc, n])

        # Log using the current round number
        fit_rows = getattr(self, "_fit_rows_cache", [])
        self.logger.log_fit_rows(self.rnd, fit_rows)
        self.logger.log_eval_rows(self.rnd, eval_rows)
        self.logger.log_avgs(self.rnd, fit_rows, eval_rows, failures)

        # visible summary
        try:
            avg_tr = sum(r[1] for r in fit_rows)/len(fit_rows) if fit_rows else float("nan")
            avg_va = sum(r[2] for r in eval_rows)/len(eval_rows) if eval_rows else float("nan")
            print(f"[server] Round {self.rnd} complete | replies={len(results)} | "
                  f"failures={len(failures)} | avg_train_loss={avg_tr:.4f} | avg_val_acc={avg_va:.4f}",
                  flush=True)
        except Exception:
            pass

        return super().aggregate_evaluate(server_round, results, failures)

    # PROJ 2-3 FUNCS:
    # Helper functions for calculating PID
    def _centroid(self, params_list):
        """List[list[np.ndarray]] -> list[np.ndarray] elementwise mean."""
        L = len(params_list[0])
        center = []
        for i in range(L):
            stacked = numpy.stack([p[i] for p in params_list], axis=0)
            center.append(stacked.mean(axis=0))
        return center

    def _model_distance(self, weights, centroid):
        return numpy.linalg.norm(numpy.concatenate([w.flatten() for w in weights])
                                 - numpy.concatenate([c.flatten() for c in centroid]))

    def _compute_pid(self, cid, dist):
        hist = self._pid_hist.get(cid, {"sum": 0.0, "prev": dist})

        # accumulate bad behavior
        # this accumulation is NORMALIZED by the # of rounds to keep evidence accumulation consistent in the simulation
        integ = hist["sum"] + (dist*(10/self.total_rounds))

        deriv = dist - hist["prev"]
        u = self.kp * dist + self.ki * integ + self.kd * deriv
        self._pid_hist[cid] = {"sum": integ, "prev": dist}
        return float(u)

def _noop_global_eval(server_round, parameters, config):
    # force Flower to schedule an evaluation phase each round
    # yes its a goofy solution
    # (returning None, {} is fine; client-side evals will still flow into aggregate_evaluate)
    return None, {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)  # results/no_attack or results/attack
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--fraction_fit", type=float, default=1.0)
    ap.add_argument("--min_clients", type=int, default=20)
    # optional knobs, defaulted to your write-up
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--kd", type=float, default=0.05)
    ap.add_argument("--pid_threshold", type=float, default=2.79)
    args = ap.parse_args()

    logger = RoundLogger(args.out_dir)
    strat = LoggedFedAvg(
        logger=logger,
        kp=args.kp, ki=args.ki, kd=args.kd, pid_threshold=args.pid_threshold,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        total_rounds=args.rounds,
        min_fit_clients=max(2, args.min_clients - 2),
        min_evaluate_clients=max(2, args.min_clients - 2),
        min_available_clients=args.min_clients,
        evaluate_fn=_noop_global_eval,  # <-- add this line
        # accept_failures=True,  # uncomment if your flwr version supports it
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strat,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )

if __name__ == "__main__":
    main()
