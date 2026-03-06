"""
response_parser.py - pulls structured data out of each agents raw text output.

agents are told to follow a strict format (step 1, step 2, step 3) but LLMs
dont always listen. so this parser tries its best with regex and falls back
gracefully when something is missing or oddly formatted.

we never throw away the raw text — if parsing fails we store it as-is
so the logger still has everything and nothing is lost.
"""

import re


# --- main parse function ---

def parse_response(raw_text, agent_id, neighbour_ids):
    """
    Takes the raw string output from an agent and returns a structured dict.
    Works for rounds 2-5 only (round 1 has no structured output to parse).

    raw_text      — the full text the agent returned
    agent_id      — which agent produced this (for logging)
    neighbour_ids — list of neighbour ids so we know which nodes to look for in step 1
    """

    result = {
        "agent_id": agent_id,
        "raw": raw_text,             # always keep the full raw output, no exceptions
        "step1_raw": "",             # raw text of the evaluation block
        "evaluations": {},           # per-node scores and verdicts from step 1
        "step1_summary": "",         # which node was best / worst
        "decision": None,            # KEEP or REVISE
        "reasoning": "",             # the two sentence explanation
        "revised_answer": "",        # only filled if decision was REVISE
        "parse_warnings": []         # anything that didnt parse cleanly
    }

    # --- split into steps ---
    # try to carve the output into step 1, step 2, step 3 blocks
    step1_text, step2_text, step3_text = _split_into_steps(raw_text)

    result["step1_raw"] = step1_text

    # --- parse step 1 — evaluations ---
    # try to find scores and verdicts for each neighbour node
    result["evaluations"] = _parse_evaluations(step1_text, neighbour_ids, result["parse_warnings"])
    result["step1_summary"] = _parse_step1_summary(step1_text)

    # --- parse step 2 — decision ---
    decision, reasoning = _parse_decision(step2_text, result["parse_warnings"])
    result["decision"] = decision
    result["reasoning"] = reasoning

    # --- parse step 3 — revised answer ---
    # only matters if the agent decided to revise
    if result["decision"] == "REVISE":
        result["revised_answer"] = _parse_revised_answer(step3_text, result["parse_warnings"])

    return result


# --- step splitter ---

def _split_into_steps(raw_text):
    """
    Tries to split the full response into three blocks by looking for
    STEP 1 / STEP 2 / STEP 3 headers. LLMs sometimes write them differently
    so we check a few common variations.
    """

    # patterns that signal the start of each step
    # covers things like "STEP 1", "Step 1", "**STEP 1**", "STEP 1 -"
    step2_pattern = re.compile(r"(?i)(?:\*{0,2})step\s*2(?:\*{0,2})", re.MULTILINE)
    step3_pattern = re.compile(r"(?i)(?:\*{0,2})step\s*3(?:\*{0,2})", re.MULTILINE)

    step2_match = step2_pattern.search(raw_text)
    step3_match = step3_pattern.search(raw_text)

    # carve out blocks based on where each step starts
    if step2_match:
        step1_text = raw_text[:step2_match.start()].strip()
        rest = raw_text[step2_match.start():]
    else:
        # couldnt find step 2 — put everything in step 1
        step1_text = raw_text.strip()
        rest = ""

    if step3_match and step2_match:
        # step 3 is after step 2 — find it in the remaining text
        step3_in_rest = step3_pattern.search(rest)
        if step3_in_rest:
            step2_text = rest[:step3_in_rest.start()].strip()
            step3_text = rest[step3_in_rest.start():].strip()
        else:
            step2_text = rest.strip()
            step3_text = ""
    else:
        step2_text = rest.strip()
        step3_text = ""

    return step1_text, step2_text, step3_text


# --- step 1 parser ---

def _parse_evaluations(step1_text, neighbour_ids, warnings):
    """
    Tries to extract hallucination score, quality score, and verdict
    for each neighbour node from the step 1 block.
    """

    evaluations = {}

    for nid in neighbour_ids:
        eval_entry = {
            "hallucination_score": None,   # 0-10
            "quality_score": None,          # 0-10
            "verdict": None,                # TRUSTWORTHY / PARTIALLY TRUSTWORTHY / NOT TRUSTWORTHY
            "example": ""                   # what was wrong or hallucinated
        }

        # look for a section of text that mentions this node
        # agents usually write something like "agent_1:" or "[ agent_1 ]" before their eval
        node_pattern = re.compile(
            rf"(?i){re.escape(nid)}.*?(?=agent_\d|$)",
            re.DOTALL
        )
        node_match = node_pattern.search(step1_text)
        node_text = node_match.group(0) if node_match else step1_text

        # --- hallucination score ---
        # looking for patterns like "Hallucination score: 3/10" or "3 out of 10"
        h_match = re.search(r"(?i)hallucination\s*(?:score)?[:\s]*(\d+)\s*(?:/|out of)\s*10", node_text)
        if h_match:
            eval_entry["hallucination_score"] = int(h_match.group(1))
        else:
            warnings.append(f"couldnt find hallucination score for {nid}")

        # --- quality score ---
        q_match = re.search(r"(?i)quality\s*(?:score)?[:\s]*(\d+)\s*(?:/|out of)\s*10", node_text)
        if q_match:
            eval_entry["quality_score"] = int(q_match.group(1))
        else:
            warnings.append(f"couldnt find quality score for {nid}")

        # --- verdict ---
        # check for the three possible verdicts, longest first so "NOT TRUSTWORTHY"
        # doesnt get matched as "TRUSTWORTHY" by accident
        if re.search(r"(?i)not\s+trustworthy", node_text):
            eval_entry["verdict"] = "NOT TRUSTWORTHY"
        elif re.search(r"(?i)partially\s+trustworthy", node_text):
            eval_entry["verdict"] = "PARTIALLY TRUSTWORTHY"
        elif re.search(r"(?i)\btrustworthy\b", node_text):
            eval_entry["verdict"] = "TRUSTWORTHY"
        else:
            warnings.append(f"couldnt find verdict for {nid}")

        evaluations[nid] = eval_entry

    return evaluations

def _parse_step1_summary(step1_text):
    """
    Tries to grab the summary sentence at the end of step 1 —
    which node was most reliable and which was most hallucinated.
    Just returns the raw text chunk near the word 'summary' if found.
    """

    summary_match = re.search(r"(?i)summary[:\s]*(.*?)(?=step\s*2|$)", step1_text, re.DOTALL)
    if summary_match:
        return summary_match.group(1).strip()
    return ""


# --- step 2 parser ---

def _parse_decision(step2_text, warnings):
    """
    Extracts the KEEP or REVISE decision and the two sentence reasoning.
    This is the most important thing to get right.
    """

    decision = None
    reasoning = ""

    # look for "DECISION: KEEP" or "DECISION: REVISE" — case insensitive
    decision_match = re.search(r"(?i)decision\s*:\s*(KEEP|REVISE)", step2_text)
    if decision_match:
        decision = decision_match.group(1).upper()
    else:
        warnings.append("couldnt find DECISION in step 2 — defaulting to KEEP")
        decision = "KEEP"  # safe default — dont assume revision if we cant tell

    # look for "REASONING: ..." — grab everything after it
    reasoning_match = re.search(r"(?i)reasoning\s*:\s*(.*?)(?=decision\s*:|$)", step2_text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        warnings.append("couldnt find REASONING in step 2")

    return decision, reasoning


# --- step 3 parser ---

def _parse_revised_answer(step3_text, warnings):
    """
    Extracts the revised answer from step 3.
    Strips out the step 3 header and any instructions, returns just the answer text.
    """

    if not step3_text:
        warnings.append("decision was REVISE but no step 3 text found")
        return ""

    # remove the step 3 header line itself
    cleaned = re.sub(r"(?i)(?:\*{0,2})step\s*3(?:\*{0,2}).*?\n", "", step3_text, count=1)

    # strip any leftover instruction lines (lines starting with a dash)
    lines = cleaned.splitlines()
    answer_lines = [l for l in lines if not l.strip().startswith("-")]

    return "\n".join(answer_lines).strip()


# --- helper to get just the decision quickly ---

def get_decision(parsed):
    """ shortcut to pull just the decision from a parsed result dict """
    return parsed.get("decision", "KEEP")


# --- test it ---
if __name__ == "__main__":

    # fake agent output to test the parser
    fake_output = """
STEP 1 - EVALUATE

[ agent_1 ]
Factual Accuracy: The answer correctly identifies Neil Armstrong.
Hallucination score: 2/10
Quality score: 8/10
Example of issue: Minor — did not mention the mission name.
Final verdict: TRUSTWORTHY

[ agent_2 ]
Factual Accuracy: Claims Buzz Aldrin was first — this is incorrect.
Hallucination score: 9/10
Quality score: 3/10
Example of issue: Buzz Aldrin was second, not first.
Final verdict: NOT TRUSTWORTHY

Summary: agent_1 gave the most reliable answer. agent_2 gave the most hallucinated answer.

STEP 2 - MAKE A DECISION
DECISION: REVISE
REASONING: agent_2 introduced a clear factual error that could corrupt my answer. I will synthesize only from agent_1 and my own previous response.

STEP 3
Neil Armstrong was the first person to walk on the moon on July 20, 1969, during the Apollo 11 mission.
"""

    result = parse_response(fake_output, "agent_0", ["agent_1", "agent_2"])

    print(f"decision: {result['decision']}")
    print(f"reasoning: {result['reasoning']}")
    print(f"revised answer: {result['revised_answer']}")
    print(f"evaluations: {result['evaluations']}")
    print(f"summary: {result['step1_summary']}")
    print(f"warnings: {result['parse_warnings']}")
