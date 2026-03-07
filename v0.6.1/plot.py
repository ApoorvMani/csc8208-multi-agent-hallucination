"""
plot.py - plot hallucination results from your manual annotations
CSC8208 Multi-Agent Hallucination Detection — Newcastle University
v0.6.1

HOW TO USE:
  1. Run experiment.py — this creates results/annotations.json
  2. Open results/annotations.json
  3. For every agent in every round, set "hallucinating": true or false
     (null means unchecked — those rounds will be skipped in the plot)
  4. Run this file:
       python plot.py

Output:
  results/hallucination_plot.png — heatmap grid (agents x rounds)
  results/hallucination_rate.png — line chart of hallucination rate per round
"""

import os                          # file paths
import json                        # reading annotation file
import numpy as np                 # grid arrays
import matplotlib.pyplot as plt    # plotting
import matplotlib.patches as mpt   # legend patches

# --- paths ---

RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results")   # results folder
ANNOTATION_FILE = os.path.join(RESULTS_DIR, "annotations.json")        # filled-in annotations

# --- style ---

plt.style.use("dark_background")   # dark theme

BG     = "#1e1e2e"   # background colour
GREEN  = "#06d6a0"   # correct — matches ground truth
RED    = "#ef476f"   # hallucinating — contradicts ground truth
GREY   = "#555577"   # null — not yet annotated


def load_annotations():
    """Load the annotation file. Exit with a clear message if it's missing."""
    if not os.path.exists(ANNOTATION_FILE):                      # check file exists
        print(f"[error] annotation file not found: {ANNOTATION_FILE}")
        print("  run experiment.py first, then fill in the annotations.")
        exit(1)                                                  # stop here

    with open(ANNOTATION_FILE, "r") as f:   # open file
        data = json.load(f)                  # parse JSON

    print(f"[loaded] {ANNOTATION_FILE}")
    print(f"  question : {data['question']}")
    return data                              # return full annotation dict


def build_grid(data):
    """
    Build a 2D numpy grid from the annotations.

    rows = agents, columns = rounds
    cell values: 1.0 = hallucinating, 0.0 = correct, 0.5 = null (unchecked)

    Returns:
      grid      — numpy array (n_agents x n_rounds)
      agent_ids — list of agent id strings (row labels)
      rounds    — list of round numbers (column labels)
    """
    agent_ids = list(data["rounds"][0]["agents"].keys())   # agent ids from first round
    rounds    = [r["round"] for r in data["rounds"]]       # round numbers

    n_agents = len(agent_ids)     # number of rows
    n_rounds = len(rounds)        # number of columns

    grid = np.full((n_agents, n_rounds), 0.5)   # default 0.5 = unchecked (grey)

    for col, round_entry in enumerate(data["rounds"]):       # iterate over rounds (columns)
        for row, agent_id in enumerate(agent_ids):           # iterate over agents (rows)
            h = round_entry["agents"][agent_id]["hallucinating"]   # true / false / null
            if h is True:                                    # hallucinating
                grid[row, col] = 1.0
            elif h is False:                                 # correct
                grid[row, col] = 0.0
            # else: null → stays 0.5 (grey)

    return grid, agent_ids, rounds


def plot_heatmap(grid, agent_ids, rounds, data):
    """
    Draw a heatmap grid: rows = agents, columns = rounds.
    Red = hallucinating, Green = correct, Grey = unchecked.
    Saved to results/hallucination_plot.png.
    """
    n_agents = len(agent_ids)   # number of rows
    n_rounds = len(rounds)      # number of columns

    fig, ax = plt.subplots(figsize=(max(14, n_rounds * 0.18), max(4, n_agents * 1.5)))   # size scales with rounds
    fig.patch.set_facecolor(BG)   # dark background
    ax.set_facecolor(BG)          # dark axes background

    for row in range(n_agents):        # draw one row per agent
        for col in range(n_rounds):    # draw one cell per round

            val = grid[row, col]                     # 0.0, 0.5, or 1.0
            if val == 1.0:                           # hallucinating
                color = RED
                label = "H"
            elif val == 0.0:                         # correct
                color = GREEN
                label = "✓"
            else:                                    # unchecked
                color = GREY
                label = "?"

            rect = plt.Rectangle(                    # draw filled cell
                [col, row], 1, 1,
                facecolor=color,
                edgecolor=BG,
                linewidth=1.5
            )
            ax.add_patch(rect)                       # add cell to axes

            # only show text label if there are <= 50 rounds (otherwise too crowded)
            if n_rounds <= 50:
                ax.text(                             # text label inside cell
                    col + 0.5, row + 0.5, label,
                    ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold"
                )

    # axis limits and ticks
    ax.set_xlim(0, n_rounds)                         # x spans all rounds
    ax.set_ylim(0, n_agents)                         # y spans all agents

    # x-axis: show every 10th round label to avoid crowding
    tick_cols   = [c for c in range(n_rounds) if (c % 10 == 0 or c == n_rounds - 1)]   # every 10th
    tick_labels = [str(rounds[c]) for c in tick_cols]                                   # round number as string
    ax.set_xticks([c + 0.5 for c in tick_cols])      # centre tick on cell
    ax.set_xticklabels(tick_labels, color="white", fontsize=9)

    # y-axis: one label per agent
    ax.set_yticks([r + 0.5 for r in range(n_agents)])   # centre tick on cell
    ax.set_yticklabels(agent_ids, color="white", fontsize=10)

    # title showing question and injection info
    ax.set_title(
        f"Hallucination Status — \"{data['question']}\"",
        color="white", fontsize=11, pad=12
    )
    ax.set_xlabel("Round", color="white", fontsize=10)   # x label
    ax.set_ylabel("Agent", color="white", fontsize=10)   # y label
    ax.tick_params(colors="white")                        # white tick marks

    # legend
    legend_patches = [
        mpt.Patch(facecolor=RED,   label="Hallucinating"),
        mpt.Patch(facecolor=GREEN, label="Correct"),
        mpt.Patch(facecolor=GREY,  label="Unchecked"),
    ]
    ax.legend(                                           # add legend
        handles=legend_patches,
        facecolor="#2e2e3e",
        labelcolor="white",
        edgecolor="#444466",
        loc="upper right"
    )

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "hallucination_plot.png")   # output file
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())     # save
    plt.close()                                                         # free memory
    print(f"[saved] {save_path}")


def plot_rate_over_time(grid, agent_ids, rounds, data):
    """
    Draw a line chart: x = round, y = fraction of agents hallucinating that round.
    Also draw individual agent lines.
    Saved to results/hallucination_rate.png.
    """
    n_rounds = len(rounds)   # total rounds on x-axis

    # compute overall hallucination rate per round (only over checked agents)
    overall_rate = []                                    # one value per round
    for col in range(n_rounds):                          # iterate over rounds (columns)
        col_vals = grid[:, col]                          # all agent values for this round
        checked  = col_vals[col_vals != 0.5]             # exclude unchecked (0.5)
        if len(checked) == 0:                            # no checked agents this round
            overall_rate.append(None)                    # nothing to plot
        else:
            overall_rate.append(float(np.mean(checked)))  # fraction hallucinating

    fig, ax = plt.subplots(figsize=(14, 5))   # wide plot
    fig.patch.set_facecolor(BG)               # dark background
    ax.set_facecolor(BG)                      # dark axes

    round_nums = list(rounds)   # x-axis values

    # plot individual agent lines (thin, semi-transparent)
    agent_colours = ["#a8dadc", "#e9c46a", "#f4a261"]   # one colour per agent
    for row, agent_id in enumerate(agent_ids):            # one line per agent
        agent_vals = []                                   # y-values for this agent
        for col in range(n_rounds):                       # iterate over rounds
            v = grid[row, col]                            # 0.0, 0.5, or 1.0
            agent_vals.append(None if v == 0.5 else v)   # None = skip (unchecked)

        # filter out None gaps for plotting (plot segments between non-None points)
        xs = [round_nums[c] for c in range(n_rounds) if agent_vals[c] is not None]   # x
        ys = [agent_vals[c]  for c in range(n_rounds) if agent_vals[c] is not None]  # y

        colour = agent_colours[row % len(agent_colours)]   # cycle colours
        ax.plot(xs, ys, color=colour, linewidth=1.2, alpha=0.6, label=agent_id)       # thin line

    # plot overall rate (thick white line)
    xs_all = [round_nums[c] for c in range(n_rounds) if overall_rate[c] is not None]   # x
    ys_all = [overall_rate[c] for c in range(n_rounds) if overall_rate[c] is not None]  # y
    ax.plot(xs_all, ys_all, color="white", linewidth=2.5, label="overall rate")         # thick line

    # axis formatting
    ax.set_xlim(round_nums[0], round_nums[-1])       # x from round 1 to last round
    ax.set_ylim(-0.05, 1.05)                          # y from 0 to 1 with small padding
    ax.set_xlabel("Round", color="white", fontsize=10)
    ax.set_ylabel("Hallucination Rate", color="white", fontsize=10)
    ax.set_title(
        f"Hallucination Rate Over Rounds — \"{data['question']}\"",
        color="white", fontsize=11, pad=10
    )
    ax.tick_params(colors="white")                    # white tick marks
    ax.yaxis.set_major_formatter(                     # show y as percentage
        plt.FuncFormatter(lambda y, _: f"{int(y*100)}%")
    )
    ax.legend(facecolor="#2e2e3e", labelcolor="white", edgecolor="#444466")   # legend

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "hallucination_rate.png")   # output file
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())     # save
    plt.close()                                                         # free memory
    print(f"[saved] {save_path}")


def print_stats(grid, agent_ids, rounds):
    """Print a quick summary of hallucination counts to the terminal."""
    n_rounds = len(rounds)   # total rounds
    print(f"\n{'='*50}")
    print("  HALLUCINATION SUMMARY")
    print(f"{'='*50}")

    for row, agent_id in enumerate(agent_ids):              # one row per agent
        col_vals  = grid[row, :]                            # all rounds for this agent
        checked   = col_vals[col_vals != 0.5]               # exclude unchecked
        n_checked = len(checked)                            # how many rounds you reviewed
        n_hall    = int(np.sum(checked == 1.0))             # how many were hallucinating
        pct       = (n_hall / n_checked * 100) if n_checked > 0 else 0   # percentage
        print(f"  {agent_id}: {n_hall}/{n_checked} checked rounds hallucinating ({pct:.0f}%)")

    # overall across all agents
    flat     = grid.flatten()                               # all cells
    checked  = flat[flat != 0.5]                            # exclude unchecked
    n_hall   = int(np.sum(checked == 1.0))                  # total hallucinating cells
    pct      = (n_hall / len(checked) * 100) if len(checked) > 0 else 0
    print(f"  {'─'*30}")
    print(f"  overall: {n_hall}/{len(checked)} checked ({pct:.0f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    data              = load_annotations()              # load your filled-in annotations
    grid, agent_ids, rounds = build_grid(data)          # build the 2D data grid

    print_stats(grid, agent_ids, rounds)                # print counts to terminal
    plot_heatmap(grid, agent_ids, rounds, data)         # heatmap grid plot
    plot_rate_over_time(grid, agent_ids, rounds, data)  # rate over time line chart

    print("\n  done — open results/ to see the plots")
