#!/usr/bin/env python3
"""
restores all client .pt files from clients_backup/ into clients_data/,
then clears clients_backup/ after successful restore.

usage:
restore all backed-up clients and clear backup folder
    py undo_poison_simple.py

restore specific clients
    py undo_poison_simple.py --clients client_0 client_1
"""

import os
import shutil
import argparse
from pathlib import Path

# DIRECTORY PATH Definitions
BASE_DIRECTORY = Path(__file__).resolve().parent
CLIENTS_DIRECTORY = (BASE_DIRECTORY / ".." / "clients_data").resolve()
BACKUP_DIRECTORY = (BASE_DIRECTORY / ".." / "clients_backup").resolve()


def restore_clients(client_names=None):
    if not BACKUP_DIRECTORY.exists():
        print("[ERROR] Backup directory not found: " + str(BACKUP_DIRECTORY))
        return

    if not CLIENTS_DIRECTORY.exists():
        os.makedirs(CLIENTS_DIRECTORY, exist_ok=True)

    backup_files = sorted(BACKUP_DIRECTORY.glob("client_*.pt"))

    if not backup_files:
        print("[WARN] No client files found in " + str(BACKUP_DIRECTORY))
        return

    if client_names:
        target_files = [c + ".pt" if not c.endswith(".pt") else c for c in client_names]
        backup_files = [f for f in backup_files if f.name in target_files]

    print("[INFO] Restoring " + str(len(backup_files)) + " client(s)...")

    for bf in backup_files:
        dest = CLIENTS_DIRECTORY / bf.name
        shutil.copy2(bf, dest)
        print("[OK] Restored " + str(bf.name) + str(dest))

    # After successful restore, clear backup directory
    clear_backup()


def clear_backup():
    for f in BACKUP_DIRECTORY.glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"[WARN] Could not delete {f}: {e}")
    print("[CLEANUP] Cleared all files from " + str(BACKUP_DIRECTORY))


def main():
    parser = argparse.ArgumentParser(description="Restore .pt client data from backup and clear backup folder.")
    parser.add_argument("--clients", nargs="*", help="Specific clients to restore (e.g., client_0 client_1)")
    args = parser.parse_args()
    restore_clients(args.clients)


if __name__ == "__main__":
    main()
