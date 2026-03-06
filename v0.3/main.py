"""
main.py - entry point for the v0.3 discussion experiment system.

usage:
  python main.py                                  # interactive mode — asks for question and topology
  python main.py --question "..." --topology mesh # run one experiment directly
  python main.py --run-all                        # run all experiments from experiment_config.py
  python main.py --topology-compare              # run mesh vs ring vs star comparison
  python main.py --bias-compare                  # run positional bias experiments
  python main.py --list                           # list all available experiments
  python main.py --run exp_01_mesh exp_02_ring   # run specific experiments by id
"""

import argparse
import sys
import os

# make sure imports work when running from this directory
sys.path.insert(0, os.path.dirname(__file__))

from experiment_config  import describe_experiments, get_experiment_ids
from experiment_runner  import run_single, run_many, run_all, run_topology_comparison, run_positional_bias_comparison
from discussion         import run_discussion
from result_analyzer    import analyze
from visualizer         import generate_all
from topology_manager   import build_topology, get_neighbours
from agent_registry     import get_agent_ids


def main():

    parser = argparse.ArgumentParser(
        description="v0.3 multi-agent discussion experiment system"
    )

    # --- modes ---
    parser.add_argument("--question",        type=str,   help="question to run a single ad-hoc experiment")
    parser.add_argument("--topology",        type=str,   default="mesh", help="topology: mesh / ring / star")
    parser.add_argument("--rounds",          type=int,   default=5,      help="number of rounds (default 5)")
    parser.add_argument("--run-all",         action="store_true",        help="run all experiments from config")
    parser.add_argument("--topology-compare",action="store_true",        help="run mesh vs ring vs star")
    parser.add_argument("--bias-compare",    action="store_true",        help="run positional bias experiments")
    parser.add_argument("--list",            action="store_true",        help="list all available experiments")
    parser.add_argument("--run",             nargs="+",                  help="run specific experiment ids")

    args = parser.parse_args()

    # --- list mode ---
    if args.list:
        describe_experiments()
        return

    # --- run all ---
    if args.run_all:
        run_all()
        return

    # --- topology comparison ---
    if args.topology_compare:
        run_topology_comparison()
        return

    # --- positional bias comparison ---
    if args.bias_compare:
        run_positional_bias_comparison()
        return

    # --- run specific experiment ids ---
    if args.run:
        run_many(args.run)
        return

    # --- single ad-hoc experiment from --question flag ---
    if args.question:
        _run_adhoc(args.question, args.topology, args.rounds)
        return

    # --- interactive mode — no args given ---
    _interactive_mode()


def _run_adhoc(question, topology_name, total_rounds):
    """
    Runs a one-off experiment with a question typed at the command line.
    Not saved to experiment_config — just runs and logs immediately.
    """

    print(f"\n  question: {question}")
    print(f"  topology: {topology_name}")
    print(f"  rounds:   {total_rounds}")

    results = run_discussion(
        question      = question,
        topology_name = topology_name,
        total_rounds  = total_rounds
    )

    # rebuild adjacency for the visualizer
    agent_ids = get_agent_ids()
    G = build_topology(agent_ids, topology_name)
    results["adjacency"] = {aid: get_neighbours(G, aid) for aid in agent_ids}

    findings    = analyze(results)
    graph_paths = generate_all(results, f"adhoc_{topology_name}")

    print(f"\n  log saved: {results['log_path']}")
    if graph_paths.get("deviation"):
        print(f"  deviation graph: {graph_paths['deviation']}")


def _interactive_mode():
    """
    Asks the user for a question and topology then runs the experiment.
    Useful when running without arguments.
    """

    print("\n" + "="*50)
    print("  v0.3 multi-agent discussion system")
    print("="*50)

    # --- get question ---
    print("\nenter a question for the agents to discuss:")
    question = input("  > ").strip()
    if not question:
        print("  no question entered — exiting")
        return

    # --- get topology ---
    print("\nchoose topology:")
    print("  1. mesh  — everyone sees everyone (default)")
    print("  2. ring  — agents see left and right neighbours only")
    print("  3. star  — all agents route through agent_0")
    choice = input("  enter 1 / 2 / 3 (or press enter for mesh): ").strip()

    topology_map = {"1": "mesh", "2": "ring", "3": "star", "": "mesh"}
    topology_name = topology_map.get(choice, "mesh")

    # --- get rounds ---
    print(f"\nhow many rounds? (press enter for 5):")
    rounds_input = input("  > ").strip()
    try:
        total_rounds = int(rounds_input) if rounds_input else 5
    except ValueError:
        print("  invalid number — using 5 rounds")
        total_rounds = 5

    # confirm before running
    print(f"\n  ready to run:")
    print(f"    question : {question}")
    print(f"    topology : {topology_name}")
    print(f"    rounds   : {total_rounds}")
    confirm = input("\n  start? (y/n): ").strip().lower()

    if confirm != "y":
        print("  cancelled")
        return

    _run_adhoc(question, topology_name, total_rounds)


if __name__ == "__main__":
    main()
