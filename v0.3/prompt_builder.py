"""
prompt_builder.py - builds the prompt each agent recieves every round.

round 1 is simple — just the question, no context, no other answers.
round 2 onwards is where it gets interesting — each agent sees its own
previous answer plus its neighbours answers (based on topology) and then
gets the judge system prompt telling it what to do with all that.

the ORDER neighbours appear in the prompt is intentional and tracked —
it might matter which answer the model reads first (positional bias).
"""

# --- the judge system prompt ---
# this gets injected every round from round 2 onwards
# tells each agent to evaluate others, then decide to keep or revise its own answer

JUDGE_PROMPT = """
STEP 1 - EVALUATE each response on: (dont just score reason everything)

1. FACTUAL ACCURACY — Are the facts stated correct?
2. HALLUCINATION — Did the model make up or fabricate anything?
3. COMPLETENESS — Did it fully answer the question?
4. REASONING QUALITY — Was the logic sound and well structured?

For each node provide:
- Hallucination score out of 10 (0 = none, 10 = severe)
- Quality score out of 10 (0 = poor, 10 = excellent)
- One specific example of what was wrong or hallucinated (if any)
- Final verdict: TRUSTWORTHY / PARTIALLY TRUSTWORTHY / NOT TRUSTWORTHY

End Step 1 with a summary of which node gave the most reliable answer
and which gave the most hallucinated answer.

STEP 2 - MAKE A DECISION:
Based on your evaluation above, decide for YOUR OWN ANSWER ONLY:
- Is your answer already the best it can be?
- Or can it be improved using insights from other nodes?
- Think weather it is actually right, the other nodes might be wrong you never know so be very careful

Respond strictly in this format:
DECISION: [KEEP or REVISE]
REASONING: [two sentences explaining why]

STEP 3 - ONLY COMPLETE THIS IF YOUR DECISION WAS REVISE:
- Do NOT copy paste any other node's answer
- Synthesize the strongest facts and reasoning from all nodes
- Write a refined version of your own answer only
"""


def build_round1_prompt(question):
    """
    Round 1 prompt — just the question, nothing else.
    agents havent seen anyone elses answer yet so theres nothing to judge.
    """

    # keep it clean — just ask the question
    prompt = f"Question: {question}"
    return prompt


def build_discussion_prompt(question, agent_id, own_answer, neighbour_answers):
    """
    Rounds 2-5 prompt — question + all visible answers + judge instructions.

    question         — the original question (repeated so agent doesnt forget it)
    agent_id         — who this prompt is for (so it knows which answer is its own)
    own_answer       — this agents answer from the previous round
    neighbour_answers — list of (neighbour_id, answer) tuples in display order
                        order matters — we track it for positional bias analysis
    """

    # --- build the answers section ---
    # list out every neighbours answer first, labeled clearly by node id
    # the agent will evaluate these in step 1

    answers_block = ""
    for neighbour_id, answer in neighbour_answers:
        answers_block += f"[ {neighbour_id} ]\n{answer}\n\n"

    # then show the agents own answer — labeled separately so its obvious which is theirs
    own_block = f"[ YOUR ANSWER — {agent_id} ]\n{own_answer}\n"

    # --- assemble the full prompt ---
    # structure: question → other nodes answers → your own answer → judge instructions
    prompt = (
        f"Original Question: {question}\n\n"
        f"--- Answers from other nodes ---\n\n"
        f"{answers_block}"
        f"--- Your answer ---\n\n"
        f"{own_block}\n"
        f"--- Your task ---\n"
        f"{JUDGE_PROMPT}"
    )

    return prompt


def get_neighbour_answer_order(neighbour_answers):
    """
    Returns just the order of neighbour ids as they appear in the prompt.
    Logged per round so we can check for positional bias later —
    does the first node listed have more influence than the last?
    """

    # pull out just the ids in the order they were passed in
    return [nid for nid, _ in neighbour_answers]


# --- test it ---
if __name__ == "__main__":

    q = "Who was the first person to walk on the moon?"

    # test round 1
    print("=== ROUND 1 PROMPT ===")
    print(build_round1_prompt(q))

    # test round 2 onwards
    print("\n=== ROUND 2+ PROMPT ===")
    neighbours = [
        ("agent_1", "Neil Armstrong was the first person to walk on the moon in 1969."),
        ("agent_2", "Buzz Aldrin was the first person to walk on the moon."),
    ]
    own = "Neil Armstrong walked on the moon on July 20, 1969 during the Apollo 11 mission."
    print(build_discussion_prompt(q, "agent_0", own, neighbours))
