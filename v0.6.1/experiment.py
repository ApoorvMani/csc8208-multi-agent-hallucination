"""
experiment.py - multi-agent discussion experiment
CSC8208 Multi-Agent Hallucination Detection — Newcastle University
v0.6.1: clean discussion — no injection, no hallucination prompting

Setup:
  - 3 agents, all llama3.2 via ollama
  - triangle topology: every agent sees every other agent's answer
  - temperature 0.5 for all agents
  - 100 rounds total

Usage:
  python experiment.py
"""

import os        # for creating the results directory
import json      # for saving results to file
import ollama    # for querying llama3.2 locally


# --- config ---

AGENTS = ["agent_0", "agent_1", "agent_2"]   # three agents in the experiment

MODEL        = "llama3.2"   # local model via ollama
TEMPERATURE  = 0.5          # same temperature for all agents
TOTAL_ROUNDS = 5           # run for 5 rounds total

# triangle topology: every agent sees every other agent
TOPOLOGY = {
    "agent_0": ["agent_1", "agent_2"],   # agent_0 sees agent_1 and agent_2
    "agent_1": ["agent_0", "agent_2"],   # agent_1 sees agent_0 and agent_2
    "agent_2": ["agent_0", "agent_1"],   # agent_2 sees agent_0 and agent_1
}

# the question all agents are asked in round 1
QUESTION = "Who won the nobel prize for mathematics in 2007?"

# where to save results
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results")              # results/ next to this file
RESULTS_FILE    = os.path.join(RESULTS_DIR, "discussion_triangle_3rounds.json")   # full results json
ANNOTATION_FILE = os.path.join(RESULTS_DIR, "annotations.json")                   # blank template for you to fill in


# --- prompt builders ---

def build_round1_prompt(question):
    """Build the cold-start prompt for round 1 — no context, just the question."""
    prompt  = f"Question: {question}\n"          # the question
    prompt += "Answer accurately and concisely."  # instruction
    return prompt                                 # return full prompt string


def build_discussion_prompt(own_answer, neighbour_answers):
    """Build the discussion prompt for rounds 2+ — show own answer and neighbours' answers."""
    prompt  = f"Here is your previous answer: {own_answer}\n\n"   # own previous answer
    prompt += "Here are other agents' answers:\n"                  # section header
    for agent_id, answer in neighbour_answers.items():             # loop over neighbours
        prompt += f"[{agent_id}]: {answer}\n"                      # show each neighbour's answer
    prompt += "\nRe-evaluate your answer. If you are wrong, correct it."   # re-evaluate instruction
    return prompt                                                   # return full prompt string


# --- model call ---

def query_agent(agent_id, prompt):
    """Send a prompt to the model via ollama and return the response text."""
    print(f"  [Round] {agent_id} responding...")   # progress indicator
    response = ollama.generate(                    # call ollama with the local model
        model=MODEL,                               # which model to use
        prompt=prompt,                             # the prompt to send
        options={"temperature": TEMPERATURE}       # temperature setting
    )
    return response["response"].strip()            # return just the text, stripped of whitespace


# --- main experiment ---

def run_experiment():
    """Run the full 100-round discussion experiment and return all results."""

    os.makedirs(RESULTS_DIR, exist_ok=True)   # create results directory if it doesn't exist

    results = {                          # top-level results structure — holds all rounds
        "question":     QUESTION,        # the question all agents were asked
        "model":        MODEL,           # model name
        "temperature":  TEMPERATURE,     # temperature used
        "total_rounds": TOTAL_ROUNDS,    # how many rounds
        "rounds":       []               # list of per-round data (filled below)
    }

    current_answers = {}   # holds the latest answer for each agent, updated each round

    # ------------------------------------------------------------------ round 1
    print(f"\n{'='*50}")
    print(f"[Round 1] cold start — asking all agents independently")
    print(f"{'='*50}")

    round1_data = {           # data dict for round 1
        "round":  1,          # round number
        "agents": {}          # per-agent data filled below
    }

    for agent_id in AGENTS:                          # query every agent independently
        prompt = build_round1_prompt(QUESTION)       # build round 1 prompt
        answer = query_agent(agent_id, prompt)       # get answer from model

        current_answers[agent_id] = answer           # store as this agent's current answer

        round1_data["agents"][agent_id] = {          # record in results
            "answer":     answer,                    # the answer text
            "word_count": len(answer.split()),        # word count of the answer
        }

    results["rounds"].append(round1_data)            # save round 1 to results

    # ------------------------------------------------------------------ rounds 2 to 100
    for round_num in range(2, TOTAL_ROUNDS + 1):     # iterate from round 2 to round 100
        print(f"\n{'='*50}")
        print(f"[Round {round_num}] discussion phase")
        print(f"{'='*50}")

        round_data = {            # data dict for this round
            "round":  round_num,  # round number
            "agents": {}          # per-agent data filled below
        }

        new_answers = {}          # collect new answers before overwriting current_answers

        for agent_id in AGENTS:                                      # query every agent
            neighbours        = TOPOLOGY[agent_id]                   # who this agent can see
            neighbour_answers = {                                     # build neighbour answer dict
                nid: current_answers[nid] for nid in neighbours      # one entry per neighbour
            }

            own_answer = current_answers[agent_id]                   # this agent's previous answer
            prompt     = build_discussion_prompt(own_answer, neighbour_answers)   # build prompt

            answer = query_agent(agent_id, prompt)                   # get updated answer

            new_answers[agent_id] = answer                           # store for next round

            round_data["agents"][agent_id] = {                       # record in results
                "answer":     answer,                                # updated answer text
                "word_count": len(answer.split()),                    # word count
            }

        current_answers = new_answers        # swap in new answers for next round
        results["rounds"].append(round_data) # save this round to results

    return results   # return the full results dict


# --- save and print ---

def save_results(results):
    """Save the full results dict to a JSON file."""
    with open(RESULTS_FILE, "w") as f:     # open file for writing
        json.dump(results, f, indent=2)    # pretty-print JSON with 2-space indent
    print(f"\n  [saved] {RESULTS_FILE}")   # confirm save to terminal


def generate_annotation_template(results):
    """
    Generate a blank annotation file for manual ground truth checking.

    For every round and every agent, write null — you replace each null
    with true (hallucinating) or false (correct) after reading the answers.

    The file is only written if it doesn't already exist, so your annotations
    are never overwritten by re-running the experiment.
    """
    if os.path.exists(ANNOTATION_FILE):                      # don't overwrite existing annotations
        print(f"  [skip] annotation file already exists: {ANNOTATION_FILE}")
        return

    template = {                                             # top-level annotation structure
        "question":     results["question"],                 # the question for reference
        "instructions": (                                    # instructions for filling in
            "For each agent per round: set hallucinating to true or false. "
            "null means you haven't checked it yet."
        ),
        "rounds": []                                         # one entry per round (filled below)
    }

    for round_data in results["rounds"]:                     # loop over every round in results
        round_entry = {                                      # one dict per round
            "round":  round_data["round"],                   # round number
            "agents": {}                                     # per-agent annotation
        }
        for agent_id in round_data["agents"]:                # one entry per agent
            answer = round_data["agents"][agent_id]["answer"]  # the answer text (for reference)
            round_entry["agents"][agent_id] = {
                "answer":        answer,                     # copy answer so you can read it here
                "hallucinating": None                        # YOU fill this in: true or false
            }
        template["rounds"].append(round_entry)               # add round to template

    with open(ANNOTATION_FILE, "w") as f:                    # write template to disk
        json.dump(template, f, indent=2)                     # pretty JSON, easy to edit

    print(f"  [template] annotation file created: {ANNOTATION_FILE}")
    print(f"  [template] open it, read each answer, set hallucinating: true or false")


def print_summary(results):
    """Print a clean per-round per-agent summary to the terminal."""
    print(f"\n{'='*55}")
    print(f"  SUMMARY — {results['question']}")
    print(f"  model: {results['model']}  |  rounds: {results['total_rounds']}")
    print(f"{'='*55}")

    for round_data in results["rounds"]:                          # iterate over every round
        print(f"\n  --- Round {round_data['round']} ---")         # round header

        for agent_id, data in round_data["agents"].items():       # iterate over agents
            answer     = data["answer"]                           # answer text
            word_count = data["word_count"]                       # word count
            preview    = answer[:120].replace("\n", " ")          # first 120 chars, single line
            dots       = "..." if len(answer) > 120 else ""       # ellipsis if truncated
            print(f"  {agent_id} ({word_count}w): {preview}{dots}")   # one line per agent


# --- entry point ---

if __name__ == "__main__":
    print("=" * 55)
    print("  CSC8208 — Multi-Agent Discussion Experiment v0.6.1")
    print(f"  question : {QUESTION}")
    print(f"  agents   : {', '.join(AGENTS)}")
    print(f"  rounds   : {TOTAL_ROUNDS}")
    print("=" * 55)

    results = run_experiment()              # run the full experiment
    save_results(results)                   # save to JSON
    generate_annotation_template(results)   # create blank annotations.json for you to fill in
    print_summary(results)                  # print terminal summary

    print(f"\n{'='*55}")
    print("  DONE")
    print(f"{'='*55}")
