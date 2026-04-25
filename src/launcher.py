# launcher.py
import argparse
import subprocess
import sys
import time
import os
import threading

"""
build and return a subprocess (in this case the clients)
this is set up to do unbuffered output, so you can see the program
actively working
@param script: string - path to .py program
@param args: args to pass to program
"""
def run_cmd(script, args, py=None):
    return [py, "-u", script] + args  # -u = unbuffered

"""
Realtime printer of standard output
@param prefix: label shown before printing an output line
@param proc - stdout stream from the subprocess (linked to run_cmd)
"""
def stream_lines(prefix, proc):
    """Continuously print each line from subprocess stdout in real time."""
    def _reader():
        for line in proc.stdout:
            print("[" + prefix + "] " + line.rstrip(), flush=True)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", action="store_true", help="Enable attack mode")
    ap.add_argument("--poisoned_clients", type=int, default=0, help="Number of poisoned clients")   # doesn't work
    ap.add_argument("--flip_frac", type=float, default=0.0, help="Label-flip fraction")  # doesn't work
    ap.add_argument("--rounds", type=int, default=20, help="Number of federated rounds")
    ap.add_argument("--epochs", type=int, default=1, help="Local epochs per client per round")
    ap.add_argument("--clients", type=int, default=20, help="Total number of simulated clients")
    args = ap.parse_args()

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, "server_fl.py")
    client_script = os.path.join(base_dir, "client_fl.py")
    clients_dir = os.path.normpath(os.path.join(base_dir, "..", "clients_data"))

    out_dir = os.path.join(base_dir, "..", "results",
                           "attack" if args.attack else "no_attack")
    out_dir = os.path.normpath(out_dir)

    venv_python = sys.executable
    print(f"[launcher] using python: {venv_python}")
    print(f"[launcher] results will be written to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Make the environment unbuffered with some weird code here
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # start up the server
    server_args = [
        "--out_dir", out_dir,
        "--rounds", str(args.rounds),
        "--min_clients", str(args.clients)
    ]
    server_command = run_cmd(server_script, server_args, venv_python)
    srv_p = subprocess.Popen(
        server_command,
        cwd=base_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    print("launcher] started server (pid " +  str(srv_p.pid))
    stream_lines("server", srv_p)

    # Server takes a bit to set up, so give it some time
    time.sleep(2.0)

    # start up all of our clients
    poisoned_client_set = set(range(args.poisoned_clients)) if args.attack else set()
    procs = []
    for cid in range(args.clients):
        client_args = [
            "--cid", str(cid),
            "--epochs", str(args.epochs),
            "--data_dir", clients_dir,
            "--server", "127.0.0.1:8080"
        ]
        if cid in poisoned_client_set:
            client_args += ["--malicious", "--flip_frac", str(args.flip_frac)]

        cmd = run_cmd(client_script, client_args, venv_python)
        process = subprocess.Popen(
            cmd,
            cwd=base_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        stream_lines("client " + str(cid), process)
        print("[launcher] started client" + str(cid) + " pid " +  str(process.pid))
        procs.append(process)
        time.sleep(0.1)

    # wait for clients to finish up in their separate threads
    try:
        for cid, p in enumerate(procs):
            p.wait()
            print(f"[launcher] client {cid} exited with code {p.returncode}")
    finally:
        # terminate the server
        try:
            srv_p.terminate()
            srv_p.wait(timeout=2)
        except Exception:
            srv_p.kill()

if __name__ == "__main__":
    main()
