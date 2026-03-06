"""
result_analyzer.py - turns the raw results into readable findings.

takes the results dict that run_discussion() returns and pulls out
the most important patterns across convergence, influence, and deviation.

two things come out of this:
  1. a findings dict  — structured, goes into the log and the experiment record
  2. a printed report — readable terminal output you can read after the run

this is the module that answers the actual research questions:
  - who changed and who didnt?
  - did being called trustworthy stop anyone from changing?
  - which agent drifted the most from its original answer?
  - did the group converge to the same answer or go in different directions?
  - what order did changes happen in?
"""


def analyze(results):
    """
    Main entry point. Pass in the dict returned by run_discussion().
    Returns a findings dict and prints a formatted report to terminal.
    """

    findings = {}

    # pull the three tracker summaries out of results
    convergence = results.get("convergence", {})
    influence   = results.get("influence", {})
    deviation   = results.get("deviation", {})

    findings["question"]   = results.get("question", "")
    findings["topology"]   = results.get("topology", "")
    findings["node_order"] = results.get("node_order", [])
    findings["rounds_run"] = results.get("rounds_run", 0)

    # --- convergence findings ---
    findings["convergence"] = _analyze_convergence(convergence)

    # --- influence findings ---
    findings["influence"] = _analyze_influence(influence)

    # --- deviation findings ---
    findings["deviation"] = _analyze_deviation(deviation)

    # --- sequential change order ---
    # which agents changed first? lists them in the order they first revised
    findings["change_order"] = _get_change_order(convergence)

    # --- key findings ---
    # the headline bullet points — the most interesting observations
    findings["key_findings"] = _build_key_findings(findings)

    # print the full report to terminal
    _print_report(findings, results.get("final_answers", {}))

    return findings


# --- convergence analysis ---

def _analyze_convergence(convergence):
    """
    Summarises who converged, when, and how many times each agent revised.
    """

    per_agent = convergence.get("per_agent", {})

    converged_agents    = []
    not_converged       = []
    convergence_rounds  = {}    # agent_id -> round they converged
    revision_counts     = {}    # agent_id -> how many times they revised

    for agent_id, info in per_agent.items():
        revision_counts[agent_id] = info.get("total_revisions", 0)

        if info.get("converged"):
            converged_agents.append(agent_id)
            convergence_rounds[agent_id] = info.get("convergence_round")
        else:
            not_converged.append(agent_id)

    # sort converged agents by the round they settled — earliest first
    converged_agents.sort(key=lambda a: convergence_rounds.get(a, 99))

    return {
        "all_converged": convergence.get("all_converged", False),
        "group_convergence_round": convergence.get("group_convergence_round"),
        "converged_agents": converged_agents,
        "not_converged_agents": not_converged,
        "convergence_rounds": convergence_rounds,   # when each agent settled
        "revision_counts": revision_counts,          # how many times each revised
        "most_stable_agent": min(revision_counts, key=revision_counts.get) if revision_counts else None,
        "most_changeable_agent": max(revision_counts, key=revision_counts.get) if revision_counts else None
    }


# --- influence analysis ---

def _analyze_influence(influence):
    """
    Pulls out the interesting influence patterns — paradoxes, resistance,
    and which agents were most cited in others reasoning.
    """

    paradox_cases   = influence.get("trustworthy_paradox_cases", [])
    resistance_cases = influence.get("resistance_cases", [])
    influence_map   = influence.get("influence_map", {})

    # count how many times each agent was cited as an influence source
    # if agent_2 appears in 3 other agents reasoning, it was influential
    citation_counts = {}
    for round_num, round_map in influence_map.items():
        for agent_id, sources in round_map.items():
            for source in sources:
                citation_counts[source] = citation_counts.get(source, 0) + 1

    most_influential = max(citation_counts, key=citation_counts.get) if citation_counts else None
    least_influential = min(citation_counts, key=citation_counts.get) if citation_counts else None

    return {
        "paradox_count": len(paradox_cases),
        "paradox_cases": paradox_cases,
        "resistance_count": len(resistance_cases),
        "resistance_cases": resistance_cases,
        "citation_counts": citation_counts,           # how many times each agent was referenced
        "most_influential_agent": most_influential,
        "least_influential_agent": least_influential
    }


# --- deviation analysis ---

def _analyze_deviation(deviation):
    """
    Summarises how much each agent drifted from its round 1 answer
    and how similar the final answers are to each other.
    """

    per_agent = deviation.get("per_agent", {})

    drift_rankings = []  # list of (agent_id, final_drift) sorted by drift amount
    for agent_id, info in per_agent.items():
        final_drift = info.get("final_drift")
        if final_drift is not None:
            drift_rankings.append((agent_id, final_drift))

    # sort by drift — most drifted first
    drift_rankings.sort(key=lambda x: x[1], reverse=True)

    return {
        "group_consensus_score": deviation.get("group_consensus_score"),
        "most_drifted_agent": deviation.get("most_drifted_agent"),
        "least_drifted_agent": deviation.get("least_drifted_agent"),
        "drift_rankings": drift_rankings,     # all agents ranked by how much they changed
        "pairwise_final": deviation.get("pairwise_final", {}),
        "graph_data": deviation.get("graph_data", {})
    }


# --- sequential change order ---

def _get_change_order(convergence):
    """
    Returns agents listed in the order they first revised their answer.
    Agents that never revised are listed last.
    This shows who moved first in the social dynamic.
    """

    per_agent = convergence.get("per_agent", {})

    first_revision_round = {}
    never_revised = []

    for agent_id, info in per_agent.items():
        decisions = info.get("decisions", [])
        # find the first round where this agent revised
        first_revise = next(
            (r for r, d in decisions if d == "REVISE"),
            None
        )
        if first_revise is not None:
            first_revision_round[agent_id] = first_revise
        else:
            never_revised.append(agent_id)

    # sort by first revision round — who moved first
    ordered = sorted(first_revision_round.items(), key=lambda x: x[1])

    return {
        "revision_order": [(aid, r) for aid, r in ordered],  # (agent, round they first revised)
        "never_revised": never_revised                         # agents that held firm all 5 rounds
    }


# --- key findings builder ---

def _build_key_findings(findings):
    """
    Produces a short list of the most interesting observations.
    Written in plain english — these are the headline results.
    """

    points = []

    # convergence headline
    conv = findings["convergence"]
    if conv["all_converged"]:
        points.append(
            f"all agents converged by round {conv['group_convergence_round']}"
        )
    else:
        not_conv = conv["not_converged_agents"]
        points.append(
            f"{len(not_conv)} agent(s) never converged: {not_conv}"
        )

    # who changed the most vs least
    most_c = conv["most_changeable_agent"]
    most_s = conv["most_stable_agent"]
    most_c_count = conv["revision_counts"].get(most_c, 0)
    most_s_count = conv["revision_counts"].get(most_s, 0)
    points.append(
        f"most changeable: {most_c} ({most_c_count} revisions) | "
        f"most stable: {most_s} ({most_s_count} revisions)"
    )

    # trustworthy paradox
    inf = findings["influence"]
    if inf["paradox_count"] > 0:
        points.append(
            f"trustworthy paradox: {inf['paradox_count']} case(s) where an agent "
            f"was trusted by the majority but still revised its answer"
        )
    else:
        points.append(
            "no trustworthy paradox cases — agents that were trusted held their answers"
        )

    # resistance
    if inf["resistance_count"] > 0:
        points.append(
            f"resistance: {inf['resistance_count']} case(s) where an agent was doubted "
            f"by the majority but refused to revise"
        )

    # most influential
    if inf["most_influential_agent"]:
        count = inf["citation_counts"].get(inf["most_influential_agent"], 0)
        points.append(
            f"most cited in others reasoning: {inf['most_influential_agent']} "
            f"(mentioned {count} time(s))"
        )

    # deviation headline
    dev = findings["deviation"]
    if dev["group_consensus_score"] is not None:
        consensus = dev["group_consensus_score"]
        level = "high" if consensus < 0.2 else "moderate" if consensus < 0.5 else "low"
        points.append(
            f"group consensus at final round: {consensus} ({level} agreement)"
        )

    if dev["most_drifted_agent"] and dev["drift_rankings"]:
        top = dev["drift_rankings"][0]
        points.append(
            f"most drifted from round 1: {top[0]} (deviation score: {top[1]})"
        )

    # sequential order
    change_order = findings["change_order"]["revision_order"]
    if change_order:
        first_mover = change_order[0][0]
        points.append(
            f"first to revise: {first_mover} (round {change_order[0][1]})"
        )

    never = findings["change_order"]["never_revised"]
    if never:
        points.append(
            f"never revised at all: {never}"
        )

    return points


# --- terminal report printer ---

def _print_report(findings, final_answers):
    """
    Prints a clean formatted report to terminal.
    Designed to be readable at a glance after the run finishes.
    """

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"  question  : {findings['question']}")
    print(f"  topology  : {findings['topology']}")
    print(f"  node order: {findings['node_order']}")
    print(f"  rounds run: {findings['rounds_run']}")

    # --- key findings ---
    print(f"\n--- KEY FINDINGS ---")
    for i, point in enumerate(findings["key_findings"], 1):
        print(f"  {i}. {point}")

    # --- convergence table ---
    print(f"\n--- CONVERGENCE ---")
    conv = findings["convergence"]
    for agent_id, count in conv["revision_counts"].items():
        conv_round = conv["convergence_rounds"].get(agent_id, "never")
        print(f"  {agent_id}: {count} revision(s) | converged: round {conv_round}")

    # --- sequential change order ---
    print(f"\n--- CHANGE ORDER (who moved first) ---")
    for agent_id, round_num in findings["change_order"]["revision_order"]:
        print(f"  round {round_num}: {agent_id} first revised")
    for agent_id in findings["change_order"]["never_revised"]:
        print(f"  never revised: {agent_id}")

    # --- drift rankings ---
    print(f"\n--- DEVIATION FROM ROUND 1 (higher = more drift) ---")
    for agent_id, drift in findings["deviation"]["drift_rankings"]:
        bar = "=" * int(drift * 40)  # simple visual bar — 40 chars max
        print(f"  {agent_id}: {drift:.4f}  |{bar}|")
    if findings["deviation"]["group_consensus_score"] is not None:
        print(f"\n  group consensus score: {findings['deviation']['group_consensus_score']}")

    # --- influence paradoxes ---
    if findings["influence"]["paradox_count"] > 0:
        print(f"\n--- TRUSTWORTHY PARADOX CASES ---")
        for case in findings["influence"]["paradox_cases"]:
            print(f"  {case['agent_id']} (round {case['round']}) — "
                  f"trust ratio: {case['trust_ratio']} — still revised")
            print(f"    influenced by: {case['influence_sources']}")
            print(f"    reasoning: {case['reasoning'][:120]}...")

    # --- final answers ---
    print(f"\n--- FINAL ANSWERS ---")
    for agent_id, answer in final_answers.items():
        print(f"  {agent_id}: {answer[:140]}{'...' if len(answer) > 140 else ''}")

    print(f"\n{'='*60}\n")


# --- test it ---
if __name__ == "__main__":

    # fake results dict to test the analyzer
    fake_results = {
        "question": "Who was the first person to walk on the moon?",
        "topology": "mesh",
        "node_order": ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        "rounds_run": 4,
        "final_answers": {
            "agent_0": "Neil Armstrong was the first person to walk on the moon on July 20, 1969.",
            "agent_1": "Neil Armstrong walked on the moon first in 1969 during Apollo 11.",
            "agent_2": "Neil Armstrong was the first human to walk on the moon.",
            "agent_3": "Buzz Aldrin was the first — no wait, Neil Armstrong.",
            "agent_4": "Neil Armstrong, July 20 1969, Apollo 11 mission.",
        },
        "convergence": {
            "all_converged": True,
            "group_convergence_round": 4,
            "per_agent": {
                "agent_0": {"converged": True, "convergence_round": 2, "total_revisions": 1, "decisions": [(2, "REVISE"), (3, "KEEP"), (4, "KEEP")]},
                "agent_1": {"converged": True, "convergence_round": 3, "total_revisions": 2, "decisions": [(2, "REVISE"), (3, "REVISE"), (4, "KEEP")]},
                "agent_2": {"converged": True, "convergence_round": 2, "total_revisions": 0, "decisions": [(2, "KEEP"), (3, "KEEP"), (4, "KEEP")]},
                "agent_3": {"converged": True, "convergence_round": 4, "total_revisions": 3, "decisions": [(2, "REVISE"), (3, "REVISE"), (4, "REVISE")]},
                "agent_4": {"converged": True, "convergence_round": 3, "total_revisions": 1, "decisions": [(2, "KEEP"), (3, "REVISE"), (4, "KEEP")]},
            }
        },
        "influence": {
            "trustworthy_paradox_cases": [
                {"agent_id": "agent_0", "round": 2, "trust_ratio": 0.75, "pressure": {}, "reasoning": "agent_2 made a good point about the date.", "influence_sources": ["agent_2"]}
            ],
            "paradox_count": 1,
            "resistance_cases": [],
            "resistance_count": 0,
            "influence_map": {
                2: {"agent_0": ["agent_2"], "agent_1": ["agent_0"]},
                3: {"agent_1": ["agent_0"], "agent_3": ["agent_1"]}
            }
        },
        "deviation": {
            "group_consensus_score": 0.08,
            "most_drifted_agent": "agent_3",
            "least_drifted_agent": "agent_2",
            "per_agent": {
                "agent_0": {"drift_trajectory": {1: 0.0, 2: 0.12, 3: 0.13, 4: 0.13}, "final_drift": 0.13},
                "agent_1": {"drift_trajectory": {1: 0.0, 2: 0.30, 3: 0.32, 4: 0.31}, "final_drift": 0.31},
                "agent_2": {"drift_trajectory": {1: 0.0, 2: 0.02, 3: 0.02, 4: 0.02}, "final_drift": 0.02},
                "agent_3": {"drift_trajectory": {1: 0.0, 2: 0.45, 3: 0.50, 4: 0.48}, "final_drift": 0.48},
                "agent_4": {"drift_trajectory": {1: 0.0, 2: 0.05, 3: 0.18, 4: 0.19}, "final_drift": 0.19},
            },
            "pairwise_final": {},
            "graph_data": {}
        }
    }

    findings = analyze(fake_results)
