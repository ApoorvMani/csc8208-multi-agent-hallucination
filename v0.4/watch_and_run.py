"""
watch_and_run.py - waits for the current run to finish then launches the next one

watches results/ for new json files to appear — once the current run_all.py
finishes and saves its results, this script automatically kicks off the next
run (with the updated prompt — no hallucination flagging).

usage:
  python watch_and_run.py

leave this running in a second terminal while run_all.py runs in the first.
"""

import os        # directory scanning
import time      # polling interval
import glob      # finding json files
import subprocess  # launching the next run
import sys         # python executable path

# how often to check if the run is done (seconds)
POLL_INTERVAL = 30

# results directory to watch
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# minimum number of json files we expect from a completed run_all.py
# 3 questions x 1 experiment json each + 3 interrogation jsons = 6 total
MIN_JSON_FILES = 6


def count_json_files():
    # count all json files currently saved in results/ and subfolders
    pattern = os.path.join(RESULTS_DIR, "**", "*.json")
    return len(glob.glob(pattern, recursive=True))


def get_latest_run_folder():
    # find the most recently modified run_* subfolder in results/
    if not os.path.exists(RESULTS_DIR):
        return None

    run_folders = [
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if d.startswith("run_") and os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]

    if not run_folders:
        return None

    # return the most recently modified folder
    return max(run_folders, key=os.path.getmtime)


def is_run_complete():
    # run is complete when we have at least MIN_JSON_FILES json files saved
    return count_json_files() >= MIN_JSON_FILES


def main():
    print("=" * 55)
    print("  watch_and_run.py — waiting for current run to finish")
    print(f"  watching : {RESULTS_DIR}")
    print(f"  trigger  : {MIN_JSON_FILES}+ json files saved")
    print(f"  polling every {POLL_INTERVAL}s")
    print("=" * 55)

    # wait for results directory to even exist
    while not os.path.exists(RESULTS_DIR):
        print("  results/ not created yet — waiting...")
        time.sleep(POLL_INTERVAL)

    # poll until we see enough json files
    while not is_run_complete():
        n = count_json_files()
        print(f"  [{time.strftime('%H:%M:%S')}] {n}/{MIN_JSON_FILES} json files saved — still running...")
        time.sleep(POLL_INTERVAL)

    # current run is done
    latest = get_latest_run_folder()
    print(f"\n  run complete — {count_json_files()} json files found")
    print(f"  latest run folder: {latest}")
    print("\n  launching next run (clean behavioural prompt — no hallucination flagging)...")
    print("=" * 55)

    # launch run_all.py as a subprocess using the same python interpreter
    script = os.path.join(os.path.dirname(__file__), "run_all.py")
    subprocess.run([sys.executable, script])

    print("\n  [DONE] both runs complete.")
    print(f"  results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
