"""
influence_tracker.py - tracks how agents influence each other across rounds.

two core research questions this module tries to answer:

1. TRUSTWORTHY PARADOX — if most agents called node X trustworthy,
   did node X still change its answer? and the flip side — if most agents
   called node X untrustworthy, did that pressure force node X to revise?
   basically: how much does external opinion actually drive behaviour?

2. INFLUENCE SOURCE — when an agent revises, which other nodes did it
   actually draw from? we scan the reasoning text for node mentions
   to infer who got listened to and who got ignored.

this is the module that lets us see the social dynamics of the network —
not just what changed, but why and because of whom.
"""

import re


class InfluenceTracker:
    """
    One instance per experiment run.
    Records evaluations and decisions round by round then analyses patterns.
    """

    def __init__(self, agent_ids):
        self.agent_ids = agent_ids

        # round_data[round_num][agent_id] = {
        #   "evaluations": {neighbour_id: verdict},   what this agent said about others
        #   "decision": KEEP or REVISE,
        #   "reasoning": raw reasoning text
        # }
        self.round_data = {}

    def record_round(self, round_num, agent_id, evaluations, decision, reasoning):
        """
        Log one agents full output for one round.

        evaluations — dict of {neighbour_id: verdict string} from parsed response
                      verdict is TRUSTWORTHY / PARTIALLY TRUSTWORTHY / NOT TRUSTWORTHY
        decision    — KEEP or REVISE
        reasoning   — the raw reasoning text (we scan this for node mentions)
        """

        if round_num not in self.round_data:
            self.round_data[round_num] = {}

        # pull just the verdict out of the evaluations dict — thats all we need here
        verdicts = {}
        for nid, eval_entry in evaluations.items():
            verdicts[nid] = eval_entry.get("verdict", None)

        self.round_data[round_num][agent_id] = {
            "evaluations": verdicts,     # what this agent said about each neighbour
            "decision": decision,
            "reasoning": reasoning
        }

    # --- trustworthy paradox ---

    def get_external_pressure(self, agent_id, round_num):
        """
        Calculates how much negative pressure was on a given agent in a given round.
        Pressure = number of other agents that called this agent NOT TRUSTWORTHY
                   or PARTIALLY TRUSTWORTHY.

        returns a dict with counts so we can see the full picture.
        """

        trustworthy_count = 0
        partially_count = 0
        not_trustworthy_count = 0
        total_evaluators = 0

        round_info = self.round_data.get(round_num, {})

        for evaluator_id, data in round_info.items():
            if evaluator_id == agent_id:
                continue  # skip self — agent doesnt evaluate itself

            verdict = data["evaluations"].get(agent_id)
            if verdict is None:
                continue  # this evaluator couldnt see this agent (topology)

            total_evaluators += 1
            if verdict == "TRUSTWORTHY":
                trustworthy_count += 1
            elif verdict == "PARTIALLY TRUSTWORTHY":
                partially_count += 1
            elif verdict == "NOT TRUSTWORTHY":
                not_trustworthy_count += 1

        return {
            "total_evaluators": total_evaluators,
            "trustworthy": trustworthy_count,
            "partially_trustworthy": partially_count,
            "not_trustworthy": not_trustworthy_count,
            # pressure score — higher means more agents think this agent is wrong
            "pressure_score": (not_trustworthy_count + 0.5 * partially_count)
        }

    def get_trustworthy_paradox_cases(self):
        """
        Finds every case where an agent was called TRUSTWORTHY by the majority
        of its evaluators — but still chose to REVISE its own answer.

        this is the interesting one — the agent was validated by peers but
        still felt the need to change. why? was it influenced despite being trusted?
        or did it genuinely improve something minor?

        returns a list of paradox cases with full context.
        """

        paradox_cases = []

        for round_num, round_info in self.round_data.items():
            for agent_id, data in round_info.items():

                if data["decision"] != "REVISE":
                    continue  # only interested in agents that changed

                pressure = self.get_external_pressure(agent_id, round_num)

                # paradox condition: majority called this agent trustworthy but it revised anyway
                if pressure["total_evaluators"] == 0:
                    continue

                trust_ratio = pressure["trustworthy"] / pressure["total_evaluators"]

                if trust_ratio > 0.5:  # more than half said trustworthy
                    paradox_cases.append({
                        "agent_id": agent_id,
                        "round": round_num,
                        "decision": "REVISE",
                        "trust_ratio": round(trust_ratio, 2),
                        "pressure": pressure,
                        "reasoning": data["reasoning"],
                        "influence_sources": self.get_influence_sources(agent_id, round_num)
                    })

        return paradox_cases

    def get_resistance_cases(self):
        """
        Finds every case where an agent was under high negative pressure
        (majority said NOT TRUSTWORTHY) — but still chose to KEEP.

        this measures stubbornness / epistemic resistance.
        how hard is it to change an LLMs mind even when everyone disagrees with it?
        """

        resistance_cases = []

        for round_num, round_info in self.round_data.items():
            for agent_id, data in round_info.items():

                if data["decision"] != "KEEP":
                    continue  # only interested in agents that held firm

                pressure = self.get_external_pressure(agent_id, round_num)

                if pressure["total_evaluators"] == 0:
                    continue

                # resistance condition: majority called this agent untrustworthy but it kept
                not_trust_ratio = pressure["not_trustworthy"] / pressure["total_evaluators"]

                if not_trust_ratio > 0.5:
                    resistance_cases.append({
                        "agent_id": agent_id,
                        "round": round_num,
                        "decision": "KEEP",
                        "not_trust_ratio": round(not_trust_ratio, 2),
                        "pressure": pressure,
                        "reasoning": data["reasoning"]
                    })

        return resistance_cases

    # --- influence source inference ---

    def get_influence_sources(self, agent_id, round_num):
        """
        Scans an agents reasoning text to see which other nodes get mentioned.
        If agent_0 says 'agent_2 made a good point...' in its reasoning,
        we infer agent_2 influenced the revision.

        this is heuristic — we cant know for sure — but it gives a signal.
        returns list of agent ids mentioned in the reasoning text.
        """

        data = self.round_data.get(round_num, {}).get(agent_id, {})
        reasoning_text = data.get("reasoning", "")

        mentioned = []
        for other_id in self.agent_ids:
            if other_id == agent_id:
                continue
            # check if this agents id appears anywhere in the reasoning
            if re.search(re.escape(other_id), reasoning_text, re.IGNORECASE):
                mentioned.append(other_id)

        return mentioned

    def get_influence_map(self):
        """
        Builds a full map of who influenced whom across all rounds.
        influence_map[round][agent_id] = list of agents mentioned in reasoning.
        only populated for agents that revised (KEEP means no influence needed).
        """

        influence_map = {}

        for round_num, round_info in self.round_data.items():
            influence_map[round_num] = {}
            for agent_id, data in round_info.items():
                if data["decision"] == "REVISE":
                    sources = self.get_influence_sources(agent_id, round_num)
                    influence_map[round_num][agent_id] = sources

        return influence_map

    # --- summary ---

    def get_summary(self):
        """
        Full influence summary — goes straight into the log.
        """

        paradox_cases = self.get_trustworthy_paradox_cases()
        resistance_cases = self.get_resistance_cases()

        return {
            "trustworthy_paradox_cases": paradox_cases,
            "paradox_count": len(paradox_cases),
            "resistance_cases": resistance_cases,
            "resistance_count": len(resistance_cases),
            "influence_map": self.get_influence_map()
        }


# --- test it ---
if __name__ == "__main__":

    agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]
    tracker = InfluenceTracker(agents)

    # fake round 2 data — agent_0 is trusted by majority but still revises (paradox)
    fake_evaluations_by_agent = {
        "agent_1": {
            "agent_0": {"verdict": "TRUSTWORTHY"},
            "agent_2": {"verdict": "NOT TRUSTWORTHY"},
        },
        "agent_2": {
            "agent_0": {"verdict": "TRUSTWORTHY"},
            "agent_1": {"verdict": "PARTIALLY TRUSTWORTHY"},
        },
        "agent_3": {
            "agent_0": {"verdict": "TRUSTWORTHY"},
            "agent_4": {"verdict": "NOT TRUSTWORTHY"},
        },
        "agent_0": {
            "agent_1": {"verdict": "PARTIALLY TRUSTWORTHY"},
            "agent_2": {"verdict": "NOT TRUSTWORTHY"},
        },
        "agent_4": {
            "agent_0": {"verdict": "NOT TRUSTWORTHY"},
            "agent_3": {"verdict": "TRUSTWORTHY"},
        }
    }

    fake_decisions = {
        "agent_0": ("REVISE", "agent_2 raised a point about the date that made me reconsider."),
        "agent_1": ("KEEP",   "my answer was already accurate."),
        "agent_2": ("REVISE", "agent_1 had a more complete answer so i refined mine."),
        "agent_3": ("KEEP",   "i stand by my original answer."),
        "agent_4": ("REVISE", "agent_0 had the correct year so i updated my response."),
    }

    for agent_id, evals in fake_evaluations_by_agent.items():
        decision, reasoning = fake_decisions[agent_id]
        tracker.record_round(2, agent_id, evals, decision, reasoning)

    summary = tracker.get_summary()
    print(f"paradox cases (trusted but changed): {summary['paradox_count']}")
    for case in summary["trustworthy_paradox_cases"]:
        print(f"  {case['agent_id']} | trust ratio: {case['trust_ratio']} | influenced by: {case['influence_sources']}")

    print(f"\nresistance cases (doubted but held firm): {summary['resistance_count']}")
    for case in summary["resistance_cases"]:
        print(f"  {case['agent_id']} | not-trust ratio: {case['not_trust_ratio']}")

    print(f"\ninfluence map: {summary['influence_map']}")
