"""
main.py - entry point for v0.4

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University
Experiment 1: triangle topology, same model, 10 rounds, natural hallucination detection
"""

import json      # for saving raw results
import os        # for directory creation
import datetime  # for run timestamps

from experiment import run_experiment
from visualizer import generate_all_plots


def main():
    # create results/ directory if it doesnt exist yet
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # timestamp for this run — all output files get this suffix so nothing overwrites
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 50)
    print("CSC8208 — Multi-Agent Hallucination Detection")
    print("Experiment 1: Triangle Topology, 10 Rounds")
    print("=" * 50)

    # run the full experiment
    results = run_experiment()

    # save raw data to json
    json_path = os.path.join(results_dir, f"results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DATA] saved to {json_path}")

    # generate and save all three plots
    generate_all_plots(results, results_dir, timestamp)

    print(f"\n[DONE] all files saved to {results_dir}/")


if __name__ == "__main__":
    main()
