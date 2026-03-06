"""
convergence_tracker.py - tracks when agents stop changing their answers.

convergence means an agent decided KEEP and never revised again after that.
we track this per agent and across all agents so we can answer questions like:
- did everyone eventually agree?
- who settled first, who kept changing till the end?
- at what round did the group as a whole stop moving?

round 1 has no decisions (agents just answer the question cold) so we
only record decisions from round 2 onwards.
"""


class ConvergenceTracker:
    """
    Keeps a record of every KEEP/REVISE decision made across all rounds.
    One instance per experiment run.
    """

    def __init__(self, agent_ids, total_rounds=5):
        # store all agent ids so we know who we're tracking
        self.agent_ids = agent_ids
        self.total_rounds = total_rounds

        # decisions[agent_id] = list of (round_num, decision) tuples
        # built up as the discussion runs
        self.decisions = {agent_id: [] for agent_id in agent_ids}

    def record(self, round_num, agent_id, decision):
        """
        Log one decision for one agent in one round.
        Call this after parsing each agents response.
        """

        # decision should be KEEP or REVISE — uppercase it just in case
        self.decisions[agent_id].append((round_num, decision.upper()))

    def get_decisions_for_agent(self, agent_id):
        """ returns list of (round, decision) tuples for one agent """
        return self.decisions.get(agent_id, [])

    def get_decision_at_round(self, agent_id, round_num):
        """ returns the decision a specific agent made in a specific round """
        for r, d in self.decisions[agent_id]:
            if r == round_num:
                return d
        return None  # no decision recorded for that round

    # --- convergence checks ---

    def convergence_round(self, agent_id):
        """
        Returns the round number where this agent first settled (KEEP and stayed KEEP).
        Returns None if the agent revised all the way to the last round.

        convergence = the first round where agent chose KEEP and never
        chose REVISE in any later round.
        """

        agent_decisions = self.decisions[agent_id]

        # go through rounds in order and find first stable KEEP
        for i, (round_num, decision) in enumerate(agent_decisions):
            if decision == "KEEP":
                # check that all rounds after this are also KEEP
                all_subsequent_keep = all(
                    d == "KEEP" for _, d in agent_decisions[i+1:]
                )
                if all_subsequent_keep:
                    return round_num  # this is the convergence round

        # agent never settled — kept revising or no decisions recorded
        return None

    def has_converged(self, agent_id):
        """ did this agent converge at some point? """
        return self.convergence_round(agent_id) is not None

    def all_converged(self):
        """ did every agent eventually converge? """
        return all(self.has_converged(aid) for aid in self.agent_ids)

    def group_convergence_round(self):
        """
        The round at which ALL agents had converged.
        This is the max of all individual convergence rounds.
        Returns None if any agent never converged.
        """

        if not self.all_converged():
            return None

        # latest individual convergence round = when the group as a whole settled
        return max(self.convergence_round(aid) for aid in self.agent_ids)

    def revise_count(self, agent_id):
        """ how many times did this agent revise across all rounds """
        return sum(1 for _, d in self.decisions[agent_id] if d == "REVISE")

    # --- output formats ---

    def get_decision_matrix(self):
        """
        Returns a 2D dict: decision_matrix[agent_id][round_num] = KEEP or REVISE.
        Easy to read in logs and useful for building the deviation graph later.
        """

        matrix = {}
        for agent_id in self.agent_ids:
            matrix[agent_id] = {}
            for round_num, decision in self.decisions[agent_id]:
                matrix[agent_id][round_num] = decision
        return matrix

    def get_summary(self):
        """
        Returns a full summary dict — goes straight into the log.
        Covers per-agent convergence and group-level convergence.
        """

        per_agent = {}
        for agent_id in self.agent_ids:
            per_agent[agent_id] = {
                "converged": self.has_converged(agent_id),
                "convergence_round": self.convergence_round(agent_id),
                "total_revisions": self.revise_count(agent_id),
                "decisions": self.decisions[agent_id]   # full list for the log
            }

        return {
            "all_converged": self.all_converged(),
            "group_convergence_round": self.group_convergence_round(),
            "decision_matrix": self.get_decision_matrix(),
            "per_agent": per_agent
        }

    def print_matrix(self):
        """
        Prints the decision matrix to terminal in a readable table.
        Useful for quick inspection during a run.
        """

        # figure out which rounds we have data for
        all_rounds = sorted(set(r for aid in self.agent_ids for r, _ in self.decisions[aid]))

        # header row
        header = f"{'agent':<12}" + "".join(f"  round{r}" for r in all_rounds)
        print(header)
        print("-" * len(header))

        # one row per agent
        for agent_id in self.agent_ids:
            row = f"{agent_id:<12}"
            for r in all_rounds:
                d = self.get_decision_at_round(agent_id, r)
                cell = d if d else "  ----"
                row += f"  {cell:<6}"
            conv = self.convergence_round(agent_id)
            row += f"  (converged: round {conv})" if conv else "  (never converged)"
            print(row)


# --- test it ---
if __name__ == "__main__":

    agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]
    tracker = ConvergenceTracker(agents, total_rounds=5)

    # simulate 4 rounds of decisions (rounds 2-5)
    fake_decisions = {
        "agent_0": [("REVISE", 2), ("KEEP",   3), ("KEEP", 4), ("KEEP",   5)],
        "agent_1": [("KEEP",   2), ("KEEP",   3), ("KEEP", 4), ("KEEP",   5)],
        "agent_2": [("REVISE", 2), ("REVISE", 3), ("KEEP", 4), ("KEEP",   5)],
        "agent_3": [("REVISE", 2), ("REVISE", 3), ("REVISE",4),("REVISE", 5)],
        "agent_4": [("KEEP",   2), ("REVISE", 3), ("KEEP", 4), ("KEEP",   5)],
    }

    for agent_id, rounds in fake_decisions.items():
        for decision, round_num in rounds:
            tracker.record(round_num, agent_id, decision)

    print("=== DECISION MATRIX ===")
    tracker.print_matrix()

    print("\n=== SUMMARY ===")
    summary = tracker.get_summary()
    print(f"all converged: {summary['all_converged']}")
    print(f"group convergence round: {summary['group_convergence_round']}")
    for aid, info in summary["per_agent"].items():
        print(f"  {aid}: converged={info['converged']} at round={info['convergence_round']} | revisions={info['total_revisions']}")
