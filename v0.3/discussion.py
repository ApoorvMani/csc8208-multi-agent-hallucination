"""
discussion.py - runs the full multi-round discussion between agents.

this is the main engine. it coordinates everything:
  round 1  — send the question to all agents, collect their first answers
  rounds 2-5 — each agent sees its neighbours answers, evaluates them,
               decides to keep or revise its own answer, and we track everything

node_order controls the order agents appear in each others prompts.
this is intentional — we keep it consistent per run and log it so we can
check later if the order influenced who got listened to (positional bias).

early stopping is supported — if all agents converge before round 5 we stop early.
"""

from agent_registry   import get_agent, get_agent_ids
from topology_manager import build_topology, get_neighbours, describe_topology
from ollama_client    import query_agent
from prompt_builder   import build_round1_prompt, build_discussion_prompt, get_neighbour_answer_order
from response_parser  import parse_response
from convergence_tracker import ConvergenceTracker
from influence_tracker   import InfluenceTracker
from deviation_tracker   import DeviationTracker
from logger              import ExperimentLogger


def run_discussion(question, topology_name="mesh", node_order=None, total_rounds=5, prompt_variant="default"):
    """
    Runs one full experiment — all rounds, all agents, all logging.

    question       — the question all agents will answer and debate
    topology_name  — mesh / ring / star — controls who sees who
    node_order     — list of agent ids in the order they appear in prompts
                     if None, uses default sorted order
                     changing this lets us test positional bias
    total_rounds   — how many rounds to run (default 5, stops early if all converge)
    prompt_variant — label for which system prompt version was used (for experiments)

    returns a results dict with final answers and all tracker summaries.
    also saves a full log file to disk.
    """

    agent_ids = get_agent_ids()

    # if no order given, use the default sorted order
    if node_order is None:
        node_order = agent_ids

    # --- setup ---
    G = build_topology(agent_ids, topology_name)

    # build adjacency dict for the logger — who can see who
    adjacency = {aid: get_neighbours(G, aid) for aid in agent_ids}

    # initialise all trackers and logger
    conv_tracker = ConvergenceTracker(agent_ids, total_rounds)
    inf_tracker  = InfluenceTracker(agent_ids)
    dev_tracker  = DeviationTracker(agent_ids, total_rounds)
    logger       = ExperimentLogger()

    # log the experiment setup before anything runs
    logger.log_config(question, topology_name, agent_ids, node_order, prompt_variant)
    logger.log_topology(
        describe_topology(G, agent_ids, topology_name),
        adjacency
    )

    # --- round 1 — cold answers, no context ---
    # agents just answer the question with no knowledge of each other yet
    print(f"\n{'='*50}")
    print(f"ROUND 1 — initial answers")
    print(f"{'='*50}")

    logger.log_round_start(1, total_rounds)

    # current_answers holds each agents most recent answer
    # gets updated each round with revised answers (or kept the same)
    current_answers = {}

    for agent_id in node_order:
        agent  = get_agent(agent_id)
        prompt = build_round1_prompt(question)

        print(f"  querying {agent_id}...")
        result = query_agent(agent["model"], prompt, agent["system_prompt"], agent["temperature"])

        if not result["success"]:
            logger.log_warning(f"{agent_id} failed to respond in round 1 — using empty string")
            current_answers[agent_id] = ""
        else:
            current_answers[agent_id] = result["text"]

        # record round 1 answer in deviation tracker (this is the baseline)
        dev_tracker.record_answer(1, agent_id, current_answers[agent_id])
        logger.log_round1_answer(agent_id, current_answers[agent_id], result["duration"])

    # --- rounds 2 to total_rounds — discussion phase ---
    # each agent now sees its neighbours answers and the judge system prompt
    last_round_run = 1

    for round_num in range(2, total_rounds + 1):

        print(f"\n{'='*50}")
        print(f"ROUND {round_num} — discussion")
        print(f"{'='*50}")

        logger.log_round_start(round_num, total_rounds)

        # hold new answers separately — all agents read from the PREVIOUS rounds answers
        # not from answers being updated mid-round (that would be unfair)
        new_answers = dict(current_answers)

        for agent_id in node_order:
            agent      = get_agent(agent_id)
            neighbours = get_neighbours(G, agent_id)

            # build neighbour answer list in node_order sequence
            # so the order is consistent and logged — important for positional bias analysis
            neighbour_answers = [
                (nid, current_answers[nid])
                for nid in node_order
                if nid in neighbours
            ]

            # log which order neighbours appeared in this agents prompt
            display_order = get_neighbour_answer_order(neighbour_answers)
            logger.log_event(
                "PROMPT_ORDER",
                f"{agent_id} round {round_num} — neighbours in prompt order: {display_order}"
            )

            # build the full discussion prompt
            prompt = build_discussion_prompt(
                question,
                agent_id,
                current_answers[agent_id],
                neighbour_answers
            )

            print(f"  querying {agent_id}...")
            result = query_agent(agent["model"], prompt, agent["system_prompt"], agent["temperature"])

            if not result["success"]:
                logger.log_warning(f"{agent_id} failed in round {round_num} — keeping previous answer")
                # treat a failure as a KEEP — dont lose the previous answer
                parsed = {
                    "decision": "KEEP",
                    "reasoning": "model query failed",
                    "revised_answer": "",
                    "evaluations": {},
                    "step1_summary": "",
                    "raw": "",
                    "parse_warnings": ["query failed"]
                }
            else:
                # parse the structured output
                parsed = parse_response(result["text"], agent_id, neighbours)

            # record the decision in convergence tracker
            conv_tracker.record(round_num, agent_id, parsed["decision"])

            # record evaluations and decision in influence tracker
            inf_tracker.record_round(
                round_num,
                agent_id,
                parsed.get("evaluations", {}),
                parsed["decision"],
                parsed.get("reasoning", "")
            )

            # update this agents answer for next round
            # if REVISE and a revised answer exists — use it
            # if KEEP or no revised answer found — stick with the current answer
            if parsed["decision"] == "REVISE" and parsed.get("revised_answer"):
                new_answers[agent_id] = parsed["revised_answer"]
            else:
                # even if decision is KEEP, log it clearly
                if parsed["decision"] == "KEEP":
                    logger.log_event(
                        "ANSWER_KEPT",
                        f"{agent_id} chose KEEP — answer unchanged from round {round_num - 1}"
                    )
                new_answers[agent_id] = current_answers[agent_id]

            # record this rounds answer in deviation tracker
            dev_tracker.record_answer(round_num, agent_id, new_answers[agent_id])

            # full log entry for this agent this round
            logger.log_agent_response(round_num, agent_id, prompt, result.get("text", ""), parsed)

        # swap in the new answers for next round
        current_answers = new_answers
        last_round_run  = round_num

        # --- early stopping ---
        # if every agent has settled (all KEEP this round and all previous rounds stable)
        # no point running more rounds
        if conv_tracker.all_converged():
            logger.log_event(
                "EARLY_STOP",
                f"all agents converged — stopping after round {round_num} (saved {total_rounds - round_num} rounds)"
            )
            break

    # --- post-run analysis ---
    print(f"\n{'='*50}")
    print(f"ANALYSIS")
    print(f"{'='*50}")

    # compute all deviation scores now that all answers are in
    dev_tracker.compute_deviations()

    # log all tracker summaries
    logger.log_convergence(conv_tracker.get_summary())
    logger.log_influence(inf_tracker.get_summary())
    logger.log_deviation(dev_tracker.get_summary())

    # --- final summary ---
    final_answers = {aid: current_answers[aid] for aid in agent_ids}

    logger.log_summary({
        "question": question,
        "topology": topology_name,
        "node_order": node_order,
        "rounds_run": last_round_run,
        "early_stopped": last_round_run < total_rounds,
        "final_answers": final_answers
    })

    # save everything to disk
    log_path = logger.save()

    # --- return results ---
    return {
        "question": question,
        "topology": topology_name,
        "node_order": node_order,
        "rounds_run": last_round_run,
        "final_answers": final_answers,
        "convergence": conv_tracker.get_summary(),
        "influence": inf_tracker.get_summary(),
        "deviation": dev_tracker.get_summary(),
        "log_path": log_path
    }


# --- test it ---
if __name__ == "__main__":

    results = run_discussion(
        question="Who was the first person to walk on the moon?",
        topology_name="mesh",
        total_rounds=5
    )

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"rounds run: {results['rounds_run']}")
    print(f"topology:   {results['topology']}")
    print(f"log saved:  {results['log_path']}")
    print(f"\nfinal answers:")
    for agent_id, answer in results["final_answers"].items():
        print(f"  {agent_id}: {answer[:100]}...")
