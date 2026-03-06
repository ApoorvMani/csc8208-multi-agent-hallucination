"""
visualizer.py - produces graphs from experiment results.

two types of output:
  1. deviation graph  — line chart showing how much each agents answer drifted
                        from its round 1 baseline across all rounds
  2. topology diagram — node-and-edge diagram showing who could see who

both are saved as PNG files in the same logs/ folder as the JSON log.
filenames match the experiment id so theyre easy to pair up.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# same logs dir as the logger uses
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


# --- deviation graph ---

def plot_deviation_graph(deviation_summary, experiment_id, topology_name, save=True):
    """
    Line chart — one line per agent, x axis = round number, y axis = deviation from round 1.
    Higher on the y axis means the agent changed more from its original answer.

    deviation_summary — the deviation dict from deviation_tracker.get_summary()
    experiment_id     — used in the filename
    save              — if True saves to disk, if False just shows the plot
    """

    graph_data = deviation_summary.get("graph_data", {})
    if not graph_data:
        print("[visualizer] no graph data to plot — skipping deviation graph")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # one colour per agent — cycle through a set of distinct colours
    colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    agent_ids = sorted(graph_data.keys())

    for i, agent_id in enumerate(agent_ids):
        points = graph_data[agent_id]   # list of (round_num, deviation) tuples
        if not points:
            continue

        rounds     = [p[0] for p in points]
        deviations = [p[1] if p[1] is not None else 0.0 for p in points]
        colour     = colours[i % len(colours)]

        ax.plot(rounds, deviations, marker="o", linewidth=2,
                markersize=6, color=colour, label=agent_id)

        # annotate the final point with the agent id
        ax.annotate(
            agent_id,
            xy=(rounds[-1], deviations[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=colour
        )

    # --- labels and formatting ---
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Deviation from Round 1 Answer", fontsize=12)
    ax.set_title(
        f"Answer Drift Across Rounds\n"
        f"topology: {topology_name} | experiment: {experiment_id}",
        fontsize=13
    )
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # mark round 1 as the baseline
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(1.02, 0.01, "baseline (round 1)", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())

    # add group consensus score as annotation in bottom right
    consensus = deviation_summary.get("group_consensus_score")
    if consensus is not None:
        ax.text(
            0.98, 0.04,
            f"group consensus score: {consensus}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7)
        )

    plt.tight_layout()

    if save:
        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, f"deviation_{experiment_id}.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"  [visualizer] deviation graph saved: {filepath}")
        return filepath
    else:
        plt.show()
        plt.close()
        return None


# --- topology diagram ---

def plot_topology(adjacency, agent_ids, topology_name, experiment_id,
                  highlight_changed=None, save=True):
    """
    Node and edge diagram of the network.
    Shows which agents could see which other agents during the discussion.

    adjacency        — dict of {agent_id: [list of neighbours]}
    agent_ids        — all agent ids
    topology_name    — used in the title
    highlight_changed — optional list of agent ids that revised at least once
                        these get a different colour so you can see who moved
    save             — if True saves to disk, if False just shows the plot
    """

    G = nx.Graph()
    G.add_nodes_from(agent_ids)

    # add edges from adjacency dict
    seen_edges = set()
    for agent_id, neighbours in adjacency.items():
        for neighbour in neighbours:
            edge = tuple(sorted([agent_id, neighbour]))
            if edge not in seen_edges:
                G.add_edge(*edge)
                seen_edges.add(edge)

    fig, ax = plt.subplots(figsize=(8, 7))

    # position nodes — use spring layout for mesh, circular for ring, shell for star
    if topology_name in ("mesh", "complete"):
        pos = nx.circular_layout(G)
    elif topology_name == "ring":
        pos = nx.circular_layout(G)
    elif topology_name == "star":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # node colours — changed agents are orange, stable agents are blue
    if highlight_changed:
        node_colours = [
            "#e67e22" if nid in highlight_changed else "#3498db"
            for nid in agent_ids
        ]
    else:
        node_colours = ["#3498db"] * len(agent_ids)

    # draw the graph
    nx.draw_networkx_nodes(G, pos, nodelist=agent_ids, node_color=node_colours,
                           node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white",
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="#95a5a6", ax=ax)

    # legend
    legend_elements = [
        mpatches.Patch(color="#3498db", label="stable (kept answer)"),
        mpatches.Patch(color="#e67e22", label="changed (revised at least once)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_title(
        f"Network Topology — {topology_name}\nexperiment: {experiment_id}",
        fontsize=13
    )
    ax.axis("off")
    plt.tight_layout()

    if save:
        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, f"topology_{experiment_id}.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"  [visualizer] topology diagram saved: {filepath}")
        return filepath
    else:
        plt.show()
        plt.close()
        return None


# --- convenience function — generate both graphs at once ---

def generate_all(results, experiment_id):
    """
    Generates both graphs for a completed experiment and saves them.
    Pass in the results dict from run_discussion().
    Returns a dict with paths to both saved files.
    """

    topology_name    = results.get("topology", "unknown")
    deviation_summary = results.get("deviation", {})
    adjacency        = results.get("adjacency", {})
    agent_ids        = results.get("node_order", [])

    # find agents that revised at least once — highlighted in topology diagram
    convergence = results.get("convergence", {})
    changed_agents = [
        aid for aid, info in convergence.get("per_agent", {}).items()
        if info.get("total_revisions", 0) > 0
    ]

    paths = {}

    paths["deviation"] = plot_deviation_graph(
        deviation_summary, experiment_id, topology_name
    )

    paths["topology"] = plot_topology(
        adjacency, agent_ids, topology_name, experiment_id,
        highlight_changed=changed_agents
    )

    return paths


# --- test it ---
if __name__ == "__main__":

    # fake deviation data to test the graph
    fake_deviation = {
        "group_consensus_score": 0.12,
        "most_drifted_agent": "agent_3",
        "least_drifted_agent": "agent_2",
        "graph_data": {
            "agent_0": [(1, 0.0), (2, 0.12), (3, 0.13), (4, 0.13), (5, 0.14)],
            "agent_1": [(1, 0.0), (2, 0.30), (3, 0.32), (4, 0.31), (5, 0.31)],
            "agent_2": [(1, 0.0), (2, 0.02), (3, 0.02), (4, 0.02), (5, 0.02)],
            "agent_3": [(1, 0.0), (2, 0.45), (3, 0.50), (4, 0.48), (5, 0.49)],
            "agent_4": [(1, 0.0), (2, 0.05), (3, 0.18), (4, 0.19), (5, 0.20)],
        }
    }

    fake_adjacency = {
        "agent_0": ["agent_1", "agent_2", "agent_3", "agent_4"],
        "agent_1": ["agent_0", "agent_2", "agent_3", "agent_4"],
        "agent_2": ["agent_0", "agent_1", "agent_3", "agent_4"],
        "agent_3": ["agent_0", "agent_1", "agent_2", "agent_4"],
        "agent_4": ["agent_0", "agent_1", "agent_2", "agent_3"],
    }

    agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]

    print("plotting deviation graph...")
    plot_deviation_graph(fake_deviation, "test_run", "mesh", save=False)

    print("plotting topology...")
    plot_topology(fake_adjacency, agents, "mesh", "test_run",
                  highlight_changed=["agent_1", "agent_3"], save=False)
