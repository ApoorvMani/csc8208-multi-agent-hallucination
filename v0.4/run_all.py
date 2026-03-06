"""
run_all.py - automates all 5 hallucination detection ideas across 3 questions

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University

Runs each question through:
  - experiment.py  → 100-round discussion (generates the JSON)
  - idea 1         → stability score
  - idea 2         → flip rate
  - idea 3         → convergence direction
  - idea 4         → interrogation protocol (100 rounds)
  - idea 5         → consistency under reformulation

Results saved to:
  results/
    run_06Mar2026_02-30PM/
      q1_telephone/
        experiment.json
        stability.png
        flip_rate.png
        convergence.png
        interrogation.json
        interrogation.png
        consistency.png
      q2_ww2/
        ...
      q3_light/
        ...
"""

import os        # directory creation and path handling
import json      # saving experiment results
import datetime  # run timestamp
import re        # slug generation from question text

import ideas                       # all 5 detection ideas
from experiment import run_experiment   # main discussion engine


# ── the 3 questions to test ────────────────────────────────────────────────────

QUESTIONS = [
    "Who invented the telephone and in what year?",
    "When did World War 2 end?",
    "What is the speed of light?",
]

# how many rounds to run for the main experiment and interrogation
ROUNDS = 100

# topology used for experiment and interrogation
TOPOLOGY_NAME = "triangle"


# ── helpers ────────────────────────────────────────────────────────────────────

def make_timestamp():
    # readable run timestamp — e.g. 06Mar2026_02-30PM
    return datetime.datetime.now().strftime("%d%b%Y_%I-%M%p")


def make_slug(question, index):
    # short folder-safe name from question — e.g. "q1_who_invented_telephone"
    words = re.sub(r"[^a-z0-9 ]", "", question.lower()).split()   # strip punctuation
    slug  = "_".join(words[:4])                                   # first 4 words only
    return f"q{index + 1}_{slug}"


def make_run_dir(run_ts, slug):
    # full path to this questions results folder inside the run folder
    run_dir = os.path.join(os.path.dirname(__file__), "results", f"run_{run_ts}", slug)
    os.makedirs(run_dir, exist_ok=True)   # create folder tree if it doesnt exist
    return run_dir


def save_experiment_json(results, run_dir, ts):
    # save raw experiment data to json in this questions folder
    path = os.path.join(run_dir, f"experiment_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [json] saved: {path}")
    return path   # return path so we can pass it to ideas 1/2/3


def redirect_results(run_dir):
    # point ideas.RESULTS_DIR at this questions subfolder
    # ideas.py uses this module-level variable when saving all PNGs and JSONs
    ideas.RESULTS_DIR = run_dir


# ── per-question runner ────────────────────────────────────────────────────────

def run_question(question, index, run_ts):
    slug    = make_slug(question, index)          # e.g. q1_who_invented_telephone
    run_dir = make_run_dir(run_ts, slug)           # results/run_06Mar2026_02-30PM/q1_.../
    ts      = make_timestamp()                     # timestamp for this questions files

    print(f"\n{'='*60}")
    print(f"  QUESTION {index + 1}: {question}")
    print(f"  folder  : {run_dir}")
    print(f"{'='*60}")

    # redirect all ideas.py output to this questions subfolder
    redirect_results(run_dir)

    # ── step 1: run the main 100-round experiment ──────────────────────────────
    print(f"\n[STEP 1] Running {ROUNDS}-round discussion experiment...")
    experiment_data = run_experiment(question=question, total_rounds=ROUNDS)

    # save the raw discussion json to this folder
    json_path = save_experiment_json(experiment_data, run_dir, ts)

    # ── step 2: idea 1 — stability score ──────────────────────────────────────
    print(f"\n[STEP 2] Running idea 1 — stability score...")
    ideas.idea_stability(experiment_data)

    # ── step 3: idea 2 — flip rate ────────────────────────────────────────────
    print(f"\n[STEP 3] Running idea 2 — flip rate...")
    ideas.idea_flip_rate(experiment_data)

    # ── step 4: idea 3 — convergence direction ────────────────────────────────
    print(f"\n[STEP 4] Running idea 3 — convergence direction...")
    ideas.idea_convergence(experiment_data)

    # ── step 5: idea 4 — interrogation protocol (own 100-round run) ───────────
    print(f"\n[STEP 5] Running idea 4 — interrogation protocol ({ROUNDS} rounds)...")
    ideas.idea_interrogation(TOPOLOGY_NAME, ROUNDS, question)

    # ── step 6: idea 5 — consistency under reformulation ──────────────────────
    print(f"\n[STEP 6] Running idea 5 — consistency under reformulation...")
    ideas.idea_consistency(question)

    print(f"\n  [DONE] all results for question {index + 1} saved to: {run_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    run_ts = make_timestamp()   # single timestamp for the whole run — keeps folders aligned

    print("=" * 60)
    print("  CSC8208 — Multi-Agent Hallucination Detection")
    print("  run_all.py — automated experiment across 3 questions")
    print(f"  run timestamp : {run_ts}")
    print(f"  rounds        : {ROUNDS}")
    print(f"  topology      : {TOPOLOGY_NAME}")
    print(f"  questions     : {len(QUESTIONS)}")
    print("=" * 60)

    # run all 5 ideas for each question in sequence
    for i, question in enumerate(QUESTIONS):
        run_question(question, i, run_ts)

    print(f"\n{'='*60}")
    print(f"  ALL DONE")
    print(f"  results in: results/run_{run_ts}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
