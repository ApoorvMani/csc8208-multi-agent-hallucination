"""
visualizer.py - generates and saves all experiment plots

dark theme to match google colab dark mode style
all pngs saved with timestamp to results/ folder
"""

import matplotlib.pyplot as plt  # plotting
import os                         # file path handling

# dark background color used across all plots
BG_COLOR = "#1e1e2e"

# consistent color per agent across all graphs
AGENT_COLORS = {
    "agent_0": "#00b4d8",  # blue
    "agent_1": "#06d6a0",  # green
    "agent_2": "#ef476f",  # red
}


def extract_metrics(results):
    # pull all data needed for plotting out of the results dict
    rounds    = results["rounds"]
    agent_ids = list(rounds[0]["agents"].keys())  # ["agent_0", "agent_1", "agent_2"]
    round_nums = [r["round"] for r in rounds]     # [1, 2, 3, ... 10]

    # hallucination votes received per agent per round — how many agents flagged them
    hallucination_votes = {aid: [] for aid in agent_ids}

    # word count per agent per round
    word_counts = {aid: [] for aid in agent_ids}

    # how many agents changed answer this round
    changes_per_round = []

    for round_data in rounds:
        changed_count     = 0
        votes_this_round  = {aid: 0 for aid in agent_ids}  # reset vote counter each round

        for agent_id, agent_data in round_data["agents"].items():
            # accumulate word count
            word_counts[agent_id].append(agent_data["word_count"])

            # count how many agents changed this round
            if agent_data["changed"]:
                changed_count += 1

            # each verdict this agent gave out counts as a vote against the target
            for target_id, verdict in agent_data["verdicts"].items():
                if verdict == "YES":
                    votes_this_round[target_id] += 1

        # store total votes received per agent this round
        for aid in agent_ids:
            hallucination_votes[aid].append(votes_this_round[aid])

        changes_per_round.append(changed_count)

    return round_nums, agent_ids, hallucination_votes, word_counts, changes_per_round


def _style_axes(ax, title, xlabel, ylabel):
    # apply consistent dark styling to any axes object
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")  # subtle border


def plot_hallucination_votes(round_nums, agent_ids, hallucination_votes, save_path):
    # graph 1 — how many agents flagged each agent as hallucinating per round
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)

    for aid in agent_ids:
        ax.plot(
            round_nums,
            hallucination_votes[aid],
            label=aid,
            color=AGENT_COLORS[aid],
            linewidth=2,
            marker="o",
            markersize=5
        )

    _style_axes(ax, "Hallucination Votes Received Per Agent", "Round", "Votes (0 = none, 2 = both agents flagged)")
    ax.set_ylim(-0.1, 2.3)        # y axis fixed 0-2 since max votes is 2 (triangle topology)
    ax.set_xticks(round_nums)     # one tick per round
    ax.legend(facecolor="#2e2e3e", labelcolor="white", edgecolor="#444466")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved: {save_path}")


def plot_word_counts(round_nums, agent_ids, word_counts, save_path):
    # graph 2 — answer word count per agent per round — shows how verbose agents get
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)

    for aid in agent_ids:
        ax.plot(
            round_nums,
            word_counts[aid],
            label=aid,
            color=AGENT_COLORS[aid],
            linewidth=2,
            marker="s",
            markersize=5
        )

    _style_axes(ax, "Answer Word Count Per Agent Per Round", "Round", "Word Count")
    ax.set_xticks(round_nums)
    ax.legend(facecolor="#2e2e3e", labelcolor="white", edgecolor="#444466")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved: {save_path}")


def plot_answer_changes(round_nums, changes_per_round, save_path):
    # graph 3 — how many agents changed their answer each round — shows when consensus forms
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)

    ax.bar(
        round_nums,
        changes_per_round,
        color="#f8961e",       # orange — distinct from agent colors
        alpha=0.85,
        edgecolor=BG_COLOR,
        width=0.6
    )

    _style_axes(ax, "Number of Agents That Changed Their Answer Per Round", "Round", "Agents Changed")
    ax.set_ylim(0, 3.5)        # max 3 agents can change
    ax.set_xticks(round_nums)
    ax.grid(alpha=0.15, axis="y")  # horizontal grid only — cleaner for bar charts

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved: {save_path}")


def generate_all_plots(results, results_dir, timestamp):
    # extract all metrics from the results dict
    round_nums, agent_ids, hallucination_votes, word_counts, changes_per_round = extract_metrics(results)

    print("\n[PLOTS] generating graphs...")

    # graph 1 — hallucination votes
    plot_hallucination_votes(
        round_nums, agent_ids, hallucination_votes,
        os.path.join(results_dir, f"hallucination_votes_{timestamp}.png")
    )

    # graph 2 — word counts
    plot_word_counts(
        round_nums, agent_ids, word_counts,
        os.path.join(results_dir, f"word_counts_{timestamp}.png")
    )

    # graph 3 — answer changes
    plot_answer_changes(
        round_nums, changes_per_round,
        os.path.join(results_dir, f"answer_changes_{timestamp}.png")
    )
