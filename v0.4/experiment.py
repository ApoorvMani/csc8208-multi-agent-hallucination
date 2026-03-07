"""
experiment.py - runs the full multi-round experiment

round 1 is a cold start — agents just answer the question independently
rounds 2 onwards — each agent sees its neighbours answers, re-evaluates,
and flags whether each other agent is hallucinating
"""

import ollama  # local model calls
import re      # for parsing structured responses

from config import AGENTS, TOPOLOGY, QUESTION, TOTAL_ROUNDS, SYSTEM_PROMPT


def build_round1_prompt(question):
    # round 1 — just ask the question, no context
    return f"Question: {question}"


def build_discussion_prompt(own_answer, neighbour_answers):
    # start with this agents own previous answer
    prompt = f"Here is your previous answer: {own_answer}\n\n"

    # list each neighbours answer clearly — agents just see the answers, nothing else
    prompt += "Here are other agents' answers:\n"
    for nid, answer in neighbour_answers.items():
        prompt += f"[{nid}]: {answer}\n"

    # simple re-evaluation instruction — no mention of hallucination
    prompt += "\nRe-evaluate your answer. If you are wrong, correct it.\n"

    # only ask for the updated answer — no verdicts, no judgement of others
    prompt += "\nFormat your response exactly like this:\n"
    prompt += "ANSWER: [your updated answer]\n"

    return prompt


def query_model(model, prompt, temperature):
    # send prompt to ollama and get back the raw text response
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": temperature}
    )
    return response["message"]["content"].strip()


def parse_response(raw, neighbours):
    # extract the ANSWER line — everything after ANSWER: until end of string
    answer_match = re.search(r"ANSWER:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else raw.strip()

    # no verdict parsing — agents dont judge each other, we detect from behaviour
    return answer, {}


def run_experiment(question=None, total_rounds=None):
    # allow callers to override question and rounds — falls back to config values
    q = question      if question      is not None else QUESTION
    n = total_rounds  if total_rounds  is not None else TOTAL_ROUNDS

    agent_ids = list(AGENTS.keys())  # ["agent_0", "agent_1", "agent_2"]

    # results will hold everything — rounds, answers, verdicts, metadata
    results = {
        "question":     q,
        "total_rounds": n,
        "rounds":       []
    }

    current_answers = {}  # latest answer per agent — updated each round

    # --- round 1 — cold start ---
    print(f"\n{'='*50}")
    print("ROUND 1 — initial answers")
    print(f"{'='*50}")

    round1_data = {"round": 1, "agents": {}}

    for agent_id in agent_ids:
        agent  = AGENTS[agent_id]
        prompt = build_round1_prompt(q)

        print(f"  querying {agent_id}...")
        raw = query_model(agent["model"], prompt, agent["temperature"])

        current_answers[agent_id] = raw  # store as baseline

        round1_data["agents"][agent_id] = {
            "answer":     raw,
            "verdicts":   {},             # no verdicts in round 1 — agents havent seen each other yet
            "word_count": len(raw.split()),
            "changed":    False           # round 1 is baseline — nothing to compare against
        }

        print(f"  {agent_id}: {raw[:100]}...")

    results["rounds"].append(round1_data)

    # --- rounds 2 to total_rounds — discussion phase ---
    for round_num in range(2, n + 1):
        print(f"\n{'='*50}")
        print(f"ROUND {round_num} — discussion")
        print(f"{'='*50}")

        round_data  = {"round": round_num, "agents": {}}
        new_answers = dict(current_answers)  # copy so all agents read from the same previous round

        for agent_id in agent_ids:
            agent      = AGENTS[agent_id]
            neighbours = TOPOLOGY[agent_id]  # who this agent can see

            # build the dict of neighbour answers this agent will see
            neighbour_answers = {nid: current_answers[nid] for nid in neighbours}

            prompt = build_discussion_prompt(current_answers[agent_id], neighbour_answers)

            print(f"  querying {agent_id}...")
            raw = query_model(agent["model"], prompt, agent["temperature"])

            # parse the structured output into answer + hallucination verdicts
            answer, verdicts = parse_response(raw, neighbours)

            # did this agent change its answer compared to last round?
            changed = answer.strip() != current_answers[agent_id].strip()

            new_answers[agent_id] = answer  # store updated answer for next round

            round_data["agents"][agent_id] = {
                "answer":     answer,
                "word_count": len(answer.split()),
                "changed":    changed
            }

            print(f"  {agent_id}: {'CHANGED' if changed else 'kept'}")

        current_answers = new_answers  # swap in new answers for next round
        results["rounds"].append(round_data)

    return results
