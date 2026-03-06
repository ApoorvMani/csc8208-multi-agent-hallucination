"""
experiment_runner.py - runs one or many experiments and stores all results.

takes experiment configs from experiment_config.py, runs each one through
discussion.py, collects the results, and saves a combined summary file.

you can run:
  - a single experiment by id
  - a list of specific experiments
  - all experiments in the config file

each experiment gets its own log file (from logger.py) plus graphs (from visualizer.py).
the runner also saves a combined results file at the end so you can compare
experiments side by side without opening individual log files.
"""

import json
import os
from datetime import datetime

from experiment_config import get_experiment, get_all_experiments, get_experiment_ids
from discussion        import run_discussion
from result_analyzer   import analyze
from visualizer        import generate_all

# combined results go here — alongside individual experiment logs
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def run_single(experiment_id):
    """
    Runs one experiment by its id and returns the full results.
    Saves log, graphs, and prints the analysis report.
    """

    config = get_experiment(experiment_id)

    print(f"\n{'#'*60}")
    print(f"  RUNNING: {config['id']}")
    print(f"  {config['description']}")
    print(f"{'#'*60}")

    # run the discussion
    results = run_discussion(
        question       = config["question"],
        topology_name  = config["topology"],
        node_order     = config["node_order"],
        total_rounds   = config.get("total_rounds", 5),
        prompt_variant = config.get("prompt_variant", "default")
    )

    # add the adjacency info so the visualizer can use it
    # discussion.py doesnt return it directly so we rebuild it here
    from topology_manager import build_topology, get_neighbours
    from agent_registry   import get_agent_ids
    agent_ids = get_agent_ids()
    G = build_topology(agent_ids, config["topology"])
    results["adjacency"] = {aid: get_neighbours(G, aid) for aid in agent_ids}

    # run the analysis and print the report
    findings = analyze(results)
    results["findings"] = findings

    # generate graphs
    graph_paths = generate_all(results, config["id"])
    results["graph_paths"] = graph_paths

    print(f"\n  experiment {config['id']} complete.")
    if graph_paths.get("deviation"):
        print(f"  deviation graph: {graph_paths['deviation']}")
    if graph_paths.get("topology"):
        print(f"  topology diagram: {graph_paths['topology']}")

    return results


def run_many(experiment_ids):
    """
    Runs a specific list of experiments in sequence.
    Returns a dict of {experiment_id: results}.
    Also saves a combined summary file at the end.
    """

    all_results = {}

    for exp_id in experiment_ids:
        print(f"\nstarting {exp_id}...")
        try:
            results = run_single(exp_id)
            all_results[exp_id] = results
        except Exception as e:
            print(f"  [ERROR] experiment {exp_id} failed: {e}")
            all_results[exp_id] = {"error": str(e)}

    # save combined summary
    _save_combined_summary(all_results, experiment_ids)

    return all_results


def run_all():
    """
    Runs every experiment defined in experiment_config.py.
    Good for a full batch run when you have time.
    """

    all_ids = get_experiment_ids()
    print(f"\nrunning all {len(all_ids)} experiments: {all_ids}")
    return run_many(all_ids)


def run_topology_comparison(question=None):
    """
    Shortcut — runs just the three topology experiments (mesh, ring, star)
    for easy comparison. Uses the default question unless you provide one.
    """

    topology_exp_ids = ["exp_01_mesh", "exp_02_ring", "exp_03_star"]
    print(f"\nrunning topology comparison: {topology_exp_ids}")
    return run_many(topology_exp_ids)


def run_positional_bias_comparison():
    """
    Shortcut — runs the three node order experiments to test positional bias.
    """

    bias_exp_ids = ["exp_01_mesh", "exp_04_mesh_reordered", "exp_05_mesh_shuffled"]
    print(f"\nrunning positional bias comparison: {bias_exp_ids}")
    return run_many(bias_exp_ids)


# --- combined summary saver ---

def _save_combined_summary(all_results, experiment_ids):
    """
    Saves a single JSON file that has the key findings from all experiments
    side by side. Much easier to compare than opening individual log files.
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"combined_results_{timestamp}.json"
    filepath  = os.path.join(LOG_DIR, filename)

    # pull just the key info from each result — not the full records (too big)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiments_run": experiment_ids,
        "results": {}
    }

    for exp_id, results in all_results.items():
        if "error" in results:
            summary["results"][exp_id] = {"error": results["error"]}
            continue

        findings = results.get("findings", {})

        summary["results"][exp_id] = {
            "question": results.get("question"),
            "topology": results.get("topology"),
            "node_order": results.get("node_order"),
            "rounds_run": results.get("rounds_run"),
            "key_findings": findings.get("key_findings", []),
            "convergence": {
                "all_converged": findings.get("convergence", {}).get("all_converged"),
                "group_convergence_round": findings.get("convergence", {}).get("group_convergence_round"),
                "revision_counts": findings.get("convergence", {}).get("revision_counts", {})
            },
            "deviation": {
                "group_consensus_score": findings.get("deviation", {}).get("group_consensus_score"),
                "drift_rankings": findings.get("deviation", {}).get("drift_rankings", [])
            },
            "influence": {
                "paradox_count": findings.get("influence", {}).get("paradox_count", 0),
                "resistance_count": findings.get("influence", {}).get("resistance_count", 0),
                "most_influential_agent": findings.get("influence", {}).get("most_influential_agent")
            },
            "change_order": findings.get("change_order", {}),
            "log_path": results.get("log_path"),
            "graph_paths": results.get("graph_paths", {})
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  [runner] combined summary saved: {filepath}")
    return filepath


# --- test it ---
if __name__ == "__main__":

    # run just the first experiment as a quick test
    results = run_single("exp_01_mesh")
    print(f"\ndone. rounds run: {results['rounds_run']}")
