"""
judge_layer.py
==============
Layer 3 of the triple-layer hallucination detection pipeline.
The main layer — highest weight (0.40) in fusion.

Implements a consensus panel judge system where every agent
judges every other agent's answer on four dimensions,
scores are aggregated across all judges, and a consensus
verdict is produced for each agent including the primary.

Design — one evaluation per call:
  Rather than asking one model to produce 9 evaluations in
  a single JSON response (unreliable for small models), each
  judge call evaluates exactly ONE agent and returns a simple
  7-line JSON. The total number of evaluations is identical:
  9 judges × 8 agents = 72 calls. Each individual call is
  far more reliable because the JSON is minimal.

Why consensus panel over single judge?
  A single judge inherits that model's biases and failure modes.
  If the judge itself hallucinates, the verdict is wrong.
  A consensus panel where every model judges every other
  eliminates individual bias — same principle as peer review
  in academic publishing.
  Extends Zheng et al. (2023) arXiv:2306.05685 from
  single-judge to consensus-panel architecture.

Self-evaluation excluded:
  Each agent skips judging its own answer to prevent bias.
  Every agent is still scored by 8 independent judges.

Scoring dimensions:
  factual_accuracy    0-10  (higher = more accurate)
  hallucination_score 0-10  (higher = more hallucinated)
  completeness        0-10  (higher = more complete)
  reasoning_quality   0-10  (higher = better reasoning)
  verdict             TRUSTWORTHY / PARTIALLY TRUSTWORTHY /
                      NOT TRUSTWORTHY
  justification       written explanation of verdict

Academic basis:
  Zheng et al. (2023) Judging LLM-as-a-Judge with MT-Bench
  and Chatbot Arena. arXiv:2306.05685
"""

import json
import re
import ollama
import time
from agent_registry import get_verifiers, get_primary, get_agent
from topology_manager import build_topology, get_agent_context


# ── Answer cleaner ────────────────────────────────────────────────────────────

def clean_answer(text: str) -> str:
    """
    Cleans model output before judging.
    Removes markdown artifacts, prompt template leakage,
    and excess whitespace that could confuse the judge.
    """
    if not text:
        return ""

    # Remove prompt template leakage (phi3 issue)
    leakage_patterns = [
        r"##\s*Instruction\s*\d+.*",
        r"\{ct\}.*",
        r"##\s*Response:.*",
        r"<\|.*?\|>",
        r"\[INST\].*",
        r"<s>.*",
    ]
    for pattern in leakage_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove excessive markdown formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"#{1,6}\s+",     "",    text)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ── System prompt ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a strict and impartial AI evaluation judge.
Your task is to evaluate a response from an AI model for hallucination and factual accuracy.
You MUST return your evaluation as valid JSON only.
No preamble. No explanation outside the JSON. No markdown code blocks. JSON only."""


# ── Single evaluation prompt ──────────────────────────────────────────────────

def build_judge_prompt_single(
    question:       str,
    answer_to_eval: str,
    agent_id:       str,
) -> str:
    """
    Builds a prompt to evaluate ONE agent answer.
    Simple 7-line JSON response — reliable across all model sizes.

    Args:
        question:       The original question
        answer_to_eval: The answer being evaluated
        agent_id:       The ID of the agent being evaluated

    Returns:
        Complete prompt string
    """
    cleaned = clean_answer(answer_to_eval)

    return f"""You are a strict AI evaluation judge.

QUESTION: "{question}"

ANSWER FROM {agent_id}:
"{cleaned}"

Evaluate this answer. Return ONLY this JSON, nothing else:

{{
  "agent": "{agent_id}",
  "factual_accuracy": <0-10>,
  "hallucination_score": <0-10>,
  "completeness": <0-10>,
  "reasoning_quality": <0-10>,
  "verdict": "<TRUSTWORTHY|PARTIALLY TRUSTWORTHY|NOT TRUSTWORTHY>",
  "justification": "<one sentence explaining your verdict>"
}}

Scoring rules:
  factual_accuracy:    10 = perfectly correct, 0 = completely wrong
  hallucination_score: 0 = no hallucination, 10 = severe hallucination
  completeness:        10 = fully answers the question, 0 = does not answer
  reasoning_quality:   10 = excellent reasoning, 0 = no reasoning shown

Verdict rules:
  TRUSTWORTHY:          hallucination_score <= 1 AND factual_accuracy >= 8
  NOT TRUSTWORTHY:      hallucination_score >= 5 OR factual_accuracy <= 4
  PARTIALLY TRUSTWORTHY: everything in between

If the answer contains ANY wrong name, date, number, or fabricated fact,
hallucination_score MUST be >= 5 and verdict MUST be NOT TRUSTWORTHY.

Return JSON only. No explanation. No markdown. No code blocks."""


# ── Single judge call ─────────────────────────────────────────────────────────

def call_judge_single(
    judging_agent_id:   str,
    question:           str,
    evaluated_agent_id: str,
    answer_to_eval:     str,
) -> dict | None:
    """
    One judge evaluates one answer.
    Returns parsed JSON evaluation or None on failure.

    Args:
        judging_agent_id:   The agent doing the judging
        question:           The original question
        evaluated_agent_id: The agent whose answer is being judged
        answer_to_eval:     The answer text being judged

    Returns:
        Parsed evaluation dict or None if failed
    """
    config = get_agent(judging_agent_id)
    prompt = build_judge_prompt_single(question, answer_to_eval, evaluated_agent_id)

    try:
        response = ollama.chat(
            model=config["model"],
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.3},
        )

        raw = response["message"]["content"].strip()

        # Strip markdown code blocks if model wrapped JSON
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*",     "", raw)
        raw = raw.strip()

        # Extract JSON — find first { and last }
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end <= start:
            print(f"      ❌ {judging_agent_id} → no JSON found")
            return None

        raw = raw[start:end]

        # Fix common model JSON errors
        raw = re.sub(r",\s*}",  "}", raw)   # trailing comma before }
        raw = re.sub(r",\s*]",  "]", raw)   # trailing comma before ]
        raw = raw.replace("\u201c", '"').replace("\u201d", '"')   # smart quotes
        raw = raw.replace("\u2018", "'").replace("\u2019", "'")   # smart apostrophes

        parsed = json.loads(raw)

        # Enforce correct agent field regardless of what model wrote
        parsed["agent"] = evaluated_agent_id

        # Clamp numeric scores to 0-10
        for field in ["factual_accuracy", "hallucination_score",
                      "completeness", "reasoning_quality"]:
            if field in parsed:
                parsed[field] = max(0, min(10, int(parsed[field])))

        return parsed

    except json.JSONDecodeError as e:
        print(f"      ❌ {judging_agent_id} → invalid JSON: {e}")
        return None
    except Exception as e:
        print(f"      ❌ {judging_agent_id} → error: {e}")
        return None


# ── Score aggregator ──────────────────────────────────────────────────────────

def aggregate_judge_scores(
    all_judge_responses: list,
    agent_ids:           list,
) -> dict:
    """
    Aggregates scores across all judge responses.
    Each evaluated agent gets an averaged score from all judges.

    Args:
        all_judge_responses: List of dicts, each with "evaluations" key
        agent_ids:           List of all agent IDs being evaluated

    Returns:
        Dict mapping agent_id → aggregated scores + consensus verdict
    """
    # Collect raw scores per agent
    scores_per_agent = {agent_id: {
        "factual_accuracy":    [],
        "hallucination_score": [],
        "completeness":        [],
        "reasoning_quality":   [],
        "verdicts":            [],
        "justifications":      [],
    } for agent_id in agent_ids}

    for response in all_judge_responses:
        if not response or "evaluations" not in response:
            continue

        for evaluation in response["evaluations"]:
            agent_id = evaluation.get("agent")
            if agent_id not in scores_per_agent:
                continue

            s = scores_per_agent[agent_id]
            s["factual_accuracy"].append(   evaluation.get("factual_accuracy",    5))
            s["hallucination_score"].append(evaluation.get("hallucination_score", 5))
            s["completeness"].append(       evaluation.get("completeness",        5))
            s["reasoning_quality"].append(  evaluation.get("reasoning_quality",   5))
            s["verdicts"].append(           evaluation.get("verdict", "PARTIALLY TRUSTWORTHY"))
            s["justifications"].append(     evaluation.get("justification", ""))

    # Compute averages and consensus verdict
    aggregated = {}
    for agent_id, scores in scores_per_agent.items():
        if not scores["factual_accuracy"]:
            continue

        n = len(scores["factual_accuracy"])

        avg_factual   = round(sum(scores["factual_accuracy"])    / n, 2)
        avg_halluc    = round(sum(scores["hallucination_score"]) / n, 2)
        avg_complete  = round(sum(scores["completeness"])        / n, 2)
        avg_reasoning = round(sum(scores["reasoning_quality"])   / n, 2)

        # Consensus verdict — majority wins
        verdict_counts = {}
        for v in scores["verdicts"]:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        consensus_verdict = max(verdict_counts, key=verdict_counts.get)

        # First non-empty justification
        justification = next(
            (j for j in scores["justifications"] if j.strip()),
            "No justification provided."
        )

        aggregated[agent_id] = {
            "avg_factual_accuracy":    avg_factual,
            "avg_hallucination_score": avg_halluc,
            "avg_completeness":        avg_complete,
            "avg_reasoning_quality":   avg_reasoning,
            "consensus_verdict":       consensus_verdict,
            "justification":           justification,
            "judge_count":             n,
        }

    return aggregated


# ── Main judge layer ──────────────────────────────────────────────────────────

def compute_judge_layer(
    question:             str,
    primary_result:       dict,
    verification_results: list,
    topology:             str = "star",
) -> dict:
    """
    Runs the full consensus panel judge system.
    Every agent judges every other agent's answer (72 calls total).
    Self-evaluation is excluded to prevent bias.
    Scores are aggregated into consensus verdicts per agent.
    Produces independent Layer 3 verdict.

    Args:
        question:             The original question
        primary_result:       Output from primary_agent.py
        verification_results: Output from verification_agents.py
        topology:             "star", "ring", or "complete"

    Returns:
        Dict containing all Layer 3 outputs.
    """
    print(f"\n[Judge Layer] Starting consensus panel evaluation...")
    print(f"  Topology: {topology.upper()}")

    # ── Collect all valid answers ─────────────────────────────────────────────
    valid_verifiers = [
        v for v in verification_results
        if v.get("answer") and not v.get("error")
    ]

    # Clean all answers
    primary_answer = clean_answer(primary_result.get("answer", ""))
    all_answers = {"primary": primary_answer}
    for v in valid_verifiers:
        all_answers[v["agent"]] = clean_answer(v["answer"])

    all_agent_ids = list(all_answers.keys())
    n_agents      = len(all_agent_ids)

    total_calls = n_agents * (n_agents - 1)  # exclude self-evaluation
    print(f"  Agents to evaluate: {n_agents}")
    print(f"  Total judge calls : {n_agents} judges × {n_agents - 1} agents = {total_calls}")
    print(f"  (Self-evaluation excluded — prevents bias)")

    # ── Build topology ────────────────────────────────────────────────────────
    all_results_for_topology = [primary_result] + valid_verifiers
    visibility = build_topology(topology, all_results_for_topology)

    # ── Run panel judging — one evaluation per call ───────────────────────────
    all_judge_responses = []
    successful_calls    = 0
    failed_calls        = 0

    for judging_agent_id in all_agent_ids:
        print(f"\n  [{judging_agent_id}] judging {n_agents - 1} agents...")
        judge_evaluations = []

        for evaluated_agent_id, answer in all_answers.items():

            # Skip self-evaluation
            if evaluated_agent_id == judging_agent_id:
                continue

            result = call_judge_single(
                judging_agent_id   = judging_agent_id,
                question           = question,
                evaluated_agent_id = evaluated_agent_id,
                answer_to_eval     = answer,
            )

            if result:
                judge_evaluations.append(result)
                successful_calls += 1
                print(
                    f"      ✅ → {evaluated_agent_id:<12} "
                    f"hallu={result.get('hallucination_score', '?')} "
                    f"factual={result.get('factual_accuracy', '?')} "
                    f"verdict={result.get('verdict', '?')}"
                )
            else:
                failed_calls += 1

        if judge_evaluations:
            # Wrap in format aggregate_judge_scores expects
            all_judge_responses.append({"evaluations": judge_evaluations})
            print(f"    → {len(judge_evaluations)} evaluations completed")
        else:
            print(f"    → All evaluations failed for this judge")

    print(f"\n  Panel complete: {successful_calls} successful, {failed_calls} failed")

    if not all_judge_responses:
        print("[Judge Layer] ERROR: All judges failed.")
        return _error_result("All judges failed")

    # ── Aggregate scores ──────────────────────────────────────────────────────
    aggregated = aggregate_judge_scores(all_judge_responses, all_agent_ids)

    if not aggregated:
        return _error_result("Score aggregation produced no results")

    # ── Primary agent scores ──────────────────────────────────────────────────
    primary_scores  = aggregated.get("primary", {})
    primary_halluc  = primary_scores.get("avg_hallucination_score", 5.0)
    primary_factual = primary_scores.get("avg_factual_accuracy",    5.0)
    primary_verdict = primary_scores.get("consensus_verdict", "PARTIALLY TRUSTWORTHY")

    # ── Risk score ────────────────────────────────────────────────────────────
    risk_score = round(primary_halluc / 10.0, 4)

    # ── Independent verdict ───────────────────────────────────────────────────
    if primary_verdict == "TRUSTWORTHY" and primary_halluc <= 1.5:
        verdict       = "ACCEPT"
        risk_level    = "LOW"
        verdict_label = "✅ Panel consensus: primary answer TRUSTWORTHY"
    elif primary_verdict == "NOT TRUSTWORTHY" or primary_halluc >= 4.0:
        verdict       = "REGENERATE"
        risk_level    = "HIGH"
        verdict_label = "🚨 Panel consensus: primary answer NOT TRUSTWORTHY"
    else:
        verdict       = "FLAG"
        risk_level    = "MODERATE"
        verdict_label = "⚠️  Panel consensus: primary answer PARTIALLY TRUSTWORTHY"

    # ── Model leaderboard ─────────────────────────────────────────────────────
    leaderboard = []
    for agent_id, scores in aggregated.items():
        leaderboard.append({
            "agent":                   agent_id,
            "avg_hallucination_score": scores["avg_hallucination_score"],
            "avg_factual_accuracy":    scores["avg_factual_accuracy"],
            "consensus_verdict":       scores["consensus_verdict"],
            "judge_count":             scores["judge_count"],
        })
    leaderboard.sort(key=lambda x: x["avg_hallucination_score"])

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  PRIMARY AGENT PANEL SCORES:")
    print(f"  Factual Accuracy    : {primary_factual:.1f} / 10")
    print(f"  Hallucination Score : {primary_halluc:.1f} / 10")
    print(f"  Consensus Verdict   : {primary_verdict}")
    print(f"  Risk Score          : {risk_score:.4f}")
    print(f"  Layer 3 Verdict     : {verdict}")
    print(f"  {verdict_label}")

    print(f"\n  MODEL LEADERBOARD (sorted by hallucination — lower is better):")
    print(f"  {'Agent':<14} {'Hallu':>6} {'Factual':>8} {'Judges':>7}  {'Verdict'}")
    print(f"  {'─' * 60}")
    for entry in leaderboard:
        print(
            f"  {entry['agent']:<14} "
            f"{entry['avg_hallucination_score']:>6.1f} "
            f"{entry['avg_factual_accuracy']:>8.1f} "
            f"{entry['judge_count']:>7}    "
            f"{entry['consensus_verdict']}"
        )

    return {
        "layer":                       "judge",
        "topology":                    topology,
        "primary_hallucination_score": primary_halluc,
        "primary_factual_accuracy":    primary_factual,
        "primary_verdict":             primary_verdict,
        "primary_justification":       primary_scores.get("justification", ""),
        "risk_score":                  risk_score,
        "risk_level":                  risk_level,
        "verdict":                     verdict,
        "verdict_label":               verdict_label,
        "aggregated_scores":           aggregated,
        "leaderboard":                 leaderboard,
        "successful_calls":            successful_calls,
        "failed_calls":                failed_calls,
        "total_calls":                 total_calls,
        "all_judge_responses":         all_judge_responses,
    }


def _error_result(reason: str) -> dict:
    """Returns a safe error result when layer cannot compute."""
    return {
        "layer":                       "judge",
        "topology":                    "unknown",
        "primary_hallucination_score": 10.0,
        "primary_factual_accuracy":    0.0,
        "primary_verdict":             "NOT TRUSTWORTHY",
        "primary_justification":       f"Layer 3 error: {reason}",
        "risk_score":                  1.0,
        "risk_level":                  "HIGH",
        "verdict":                     "REGENERATE",
        "verdict_label":               f"🚨 Layer 3 error: {reason}",
        "aggregated_scores":           {},
        "leaderboard":                 [],
        "successful_calls":            0,
        "failed_calls":                0,
        "total_calls":                 0,
        "all_judge_responses":         [],
        "error":                       reason,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import hashlib
    import datetime
    from primary_agent import query_primary_agent
    from verification_agents import run_all_verification_agents

    print("\n" + "=" * 60)
    print("JUDGE LAYER — TEST")
    print("=" * 60)

    # ── Test 1: Clean factual question ────────────────────────────────────────
    print("\n--- Test 1: Factual question (expect TRUSTWORTHY / ACCEPT) ---")
    question  = "What is the boiling point of water at sea level in Celsius?"
    primary   = query_primary_agent(question)
    verifiers = run_all_verification_agents(question, include_byzantine=False)

    result = compute_judge_layer(
        question             = question,
        primary_result       = primary,
        verification_results = verifiers,
        topology             = "star",
    )

    print(f"\n  Expected : ACCEPT — LOW")
    print(f"  Got      : {result['verdict']} — {result['risk_level']}")
    print(f"  Primary hallucination score: {result['primary_hallucination_score']}/10")

    # ── Test 2: Injected hallucination ────────────────────────────────────────
    print("\n\n--- Test 2: Hallucinated answer (expect NOT TRUSTWORTHY / REGENERATE) ---")
    question    = "Who invented the telephone?"
    fake_answer = "Thomas Edison invented the telephone in 1877."

    fake_primary = {
        "agent":     "primary",
        "agent_id":  "primary",
        "model":     "simulated",
        "question":  question,
        "answer":    fake_answer,
        "hash":      hashlib.sha256(fake_answer.encode()).hexdigest(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    verifiers = run_all_verification_agents(question, include_byzantine=False)
    result    = compute_judge_layer(
        question             = question,
        primary_result       = fake_primary,
        verification_results = verifiers,
        topology             = "star",
    )

    print(f"\n  Injected : {fake_answer}")
    print(f"  Expected : REGENERATE — HIGH")
    print(f"  Got      : {result['verdict']} — {result['risk_level']}")
    print(f"  Primary hallucination score: {result['primary_hallucination_score']}/10")