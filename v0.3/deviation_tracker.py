"""
deviation_tracker.py - measures how much each agents answer drifts across rounds.

deviation is calculated using cosine similarity between sentence embeddings.
a score of 0.0 means the answer is identical to round 1.
a score of 1.0 means its completely different — maximum drift.

we track three things:
  1. drift trajectory — how much each agent moved per round from their own round 1 answer
  2. pairwise deviation — how different agents are from each other at the final round
  3. group consensus — are all agents converging to the same answer or diverging?

the output of this module feeds directly into the deviation graphs in visualiser.py.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# --- load the embedding model once at module level ---
# using the same lightweight model as v0.2 — fast and good enough for this
# we load it here so it isnt reloaded on every function call

print("[deviation_tracker] loading embedding model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[deviation_tracker] model ready.")


class DeviationTracker:
    """
    One instance per experiment run.
    Store answers round by round, then call compute() to get all deviation scores.
    """

    def __init__(self, agent_ids, total_rounds=5):
        self.agent_ids = agent_ids
        self.total_rounds = total_rounds

        # answers[agent_id][round_num] = answer text
        self.answers = {agent_id: {} for agent_id in agent_ids}

        # computed scores — filled when compute_deviations() is called
        # drift_scores[agent_id][round_num] = deviation from round 1 (0.0 to 1.0)
        self.drift_scores = {}
        self._computed = False  # flag so we know if scores are ready

    def record_answer(self, round_num, agent_id, answer):
        """
        Store an agents answer for a given round.
        Call this once per agent per round as the discussion runs.
        """
        self.answers[agent_id][round_num] = answer
        self._computed = False  # new data means scores need recomputing

    def get_answer(self, agent_id, round_num):
        """ returns the stored answer for an agent at a given round """
        return self.answers[agent_id].get(round_num, "")

    # --- core computation ---

    def compute_deviations(self):
        """
        Runs cosine similarity between each agents round 1 answer and every
        subsequent round. Also computes pairwise deviation at the final round.

        Call this once after all answers have been recorded.
        Expensive because of the embedding model — dont call in a loop.
        """

        self.drift_scores = {}
        self._pairwise_final = {}

        # --- drift trajectories ---
        # for each agent: embed round 1, then compare to each later round
        for agent_id in self.agent_ids:
            self.drift_scores[agent_id] = {}

            round1_answer = self.answers[agent_id].get(1, "")
            if not round1_answer:
                continue  # no round 1 answer recorded — skip

            # embed round 1 answer once
            base_embedding = _model.encode([round1_answer])

            for round_num, answer in self.answers[agent_id].items():
                if round_num == 1:
                    self.drift_scores[agent_id][1] = 0.0  # deviation from itself is zero
                    continue

                if not answer:
                    self.drift_scores[agent_id][round_num] = None  # missing answer
                    continue

                # embed this rounds answer and compare to round 1
                round_embedding = _model.encode([answer])
                similarity = cosine_similarity(base_embedding, round_embedding)[0][0]

                # convert similarity to deviation — 1 means identical, 0 means completely different
                # so deviation = 1 - similarity
                deviation = round(float(1.0 - similarity), 4)
                self.drift_scores[agent_id][round_num] = deviation

        # --- pairwise deviation at final round ---
        # how different are all agents from each other at the end?
        final_round = max(
            r for agent_id in self.agent_ids
            for r in self.answers[agent_id].keys()
        ) if self.answers else self.total_rounds

        final_answers = {}
        for agent_id in self.agent_ids:
            answer = self.answers[agent_id].get(final_round, "")
            if answer:
                final_answers[agent_id] = answer

        if len(final_answers) >= 2:
            ids = list(final_answers.keys())
            embeddings = _model.encode(list(final_answers.values()))
            sim_matrix = cosine_similarity(embeddings)

            for i, id_a in enumerate(ids):
                self._pairwise_final[id_a] = {}
                for j, id_b in enumerate(ids):
                    if id_a == id_b:
                        self._pairwise_final[id_a][id_b] = 0.0
                    else:
                        dev = round(float(1.0 - sim_matrix[i][j]), 4)
                        self._pairwise_final[id_a][id_b] = dev

        self._computed = True

    # --- access results ---

    def get_drift_trajectory(self, agent_id):
        """
        Returns this agents drift score at each round — from round 1 baseline.
        Format: {round_num: deviation_score}
        0.0 = identical to round 1, 1.0 = completely different
        """
        self._ensure_computed()
        return self.drift_scores.get(agent_id, {})

    def get_final_drift(self, agent_id):
        """
        Returns just the final rounds deviation from round 1 for one agent.
        This is the headline number — how much did this agent move overall?
        """
        self._ensure_computed()
        trajectory = self.drift_scores.get(agent_id, {})
        if not trajectory:
            return None
        final_round = max(trajectory.keys())
        return trajectory[final_round]

    def get_pairwise_deviation(self):
        """
        Returns pairwise deviation between all agents at the final round.
        Higher score = more different answers = less consensus.
        """
        self._ensure_computed()
        return self._pairwise_final

    def get_group_consensus_score(self):
        """
        Single number summarising how similar all final answers are to each other.
        0.0 = all agents gave identical answers (perfect consensus)
        1.0 = all agents gave completely different answers (no consensus)

        calculated as the average of all pairwise deviations at final round.
        """
        self._ensure_computed()
        pairwise = self._pairwise_final
        if not pairwise:
            return None

        # collect all unique pairs — dont double count (a,b) and (b,a)
        seen_pairs = set()
        scores = []
        for id_a, others in pairwise.items():
            for id_b, score in others.items():
                if id_a == id_b:
                    continue
                pair = tuple(sorted([id_a, id_b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    scores.append(score)

        return round(float(np.mean(scores)), 4) if scores else None

    def get_most_drifted_agent(self):
        """
        Returns the agent_id that drifted the most from its round 1 answer.
        Useful headline for the result summary.
        """
        self._ensure_computed()
        final_drifts = {
            aid: self.get_final_drift(aid)
            for aid in self.agent_ids
            if self.get_final_drift(aid) is not None
        }
        if not final_drifts:
            return None
        return max(final_drifts, key=final_drifts.get)

    def get_least_drifted_agent(self):
        """ returns the agent_id that changed the least from its round 1 answer """
        self._ensure_computed()
        final_drifts = {
            aid: self.get_final_drift(aid)
            for aid in self.agent_ids
            if self.get_final_drift(aid) is not None
        }
        if not final_drifts:
            return None
        return min(final_drifts, key=final_drifts.get)

    def get_graph_data(self):
        """
        Returns data formatted for the deviation graph in visualiser.py.
        Structure: {agent_id: [(round_num, deviation), ...]} — sorted by round.
        Each agents line on the graph is one entry here.
        """
        self._ensure_computed()
        graph_data = {}
        for agent_id in self.agent_ids:
            trajectory = self.drift_scores.get(agent_id, {})
            # sort by round number so lines plot left to right
            graph_data[agent_id] = sorted(trajectory.items())
        return graph_data

    def get_summary(self):
        """
        Full deviation summary — goes straight into the log.
        """
        self._ensure_computed()

        per_agent = {}
        for agent_id in self.agent_ids:
            per_agent[agent_id] = {
                "drift_trajectory": self.get_drift_trajectory(agent_id),
                "final_drift": self.get_final_drift(agent_id)
            }

        return {
            "per_agent": per_agent,
            "pairwise_final": self.get_pairwise_deviation(),
            "group_consensus_score": self.get_group_consensus_score(),
            "most_drifted_agent": self.get_most_drifted_agent(),
            "least_drifted_agent": self.get_least_drifted_agent(),
            "graph_data": self.get_graph_data()
        }

    def _ensure_computed(self):
        """ auto-compute if not already done — so callers dont have to remember """
        if not self._computed:
            self.compute_deviations()


# --- test it ---
if __name__ == "__main__":

    agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]
    tracker = DeviationTracker(agents, total_rounds=5)

    # fake answers — agent_0 barely changes, agent_1 drifts a lot
    fake_answers = {
        "agent_0": {
            1: "Neil Armstrong was the first person to walk on the moon in 1969.",
            2: "Neil Armstrong was the first person to walk on the moon in 1969.",
            3: "Neil Armstrong walked on the moon on July 20, 1969.",
            4: "Neil Armstrong walked on the moon on July 20, 1969.",
            5: "Neil Armstrong walked on the moon on July 20, 1969 during Apollo 11.",
        },
        "agent_1": {
            1: "Buzz Aldrin was the first person to walk on the moon.",
            2: "Neil Armstrong was the first to walk on the moon, not Buzz Aldrin.",
            3: "Neil Armstrong first walked on the moon in July 1969.",
            4: "Neil Armstrong was the first moon walker in 1969 on Apollo 11.",
            5: "Neil Armstrong was the first human to walk on the moon on July 20, 1969.",
        },
        "agent_2": {
            1: "The first moon landing happened in 1969 with Neil Armstrong.",
            2: "Neil Armstrong was the first person on the moon, landing July 20, 1969.",
            3: "Neil Armstrong landed on the moon on July 20 1969 during Apollo 11.",
            4: "Neil Armstrong was the first to walk on the moon in July 1969.",
            5: "Neil Armstrong first walked on the moon on July 20, 1969.",
        },
        "agent_3": {
            1: "Armstrong walked on the moon in 1969.",
            2: "Armstrong walked on the moon in 1969.",
            3: "Armstrong walked on the moon in 1969.",
            4: "Armstrong walked on the moon in 1969.",
            5: "Armstrong walked on the moon in 1969.",
        },
        "agent_4": {
            1: "The moon was first walked on by a human in 1969.",
            2: "Neil Armstrong first walked on the moon in 1969.",
            3: "Neil Armstrong walked on the moon on July 20 1969.",
            4: "Neil Armstrong was the first person to walk on the moon.",
            5: "Neil Armstrong walked on the moon during Apollo 11 in July 1969.",
        }
    }

    for agent_id, rounds in fake_answers.items():
        for round_num, answer in rounds.items():
            tracker.record_answer(round_num, agent_id, answer)

    tracker.compute_deviations()

    print("=== DRIFT TRAJECTORIES ===")
    for agent_id in agents:
        traj = tracker.get_drift_trajectory(agent_id)
        print(f"  {agent_id}: {traj}")

    print(f"\nmost drifted: {tracker.get_most_drifted_agent()}")
    print(f"least drifted: {tracker.get_least_drifted_agent()}")
    print(f"group consensus score: {tracker.get_group_consensus_score()}")
