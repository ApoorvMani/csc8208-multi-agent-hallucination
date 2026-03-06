"""
primary_agent.py
================
The Primary Agent is the first component of our Multi-Agent Hallucination
Detection Framework. It sends a question to a local LLM (Mistral via Ollama)
and returns a structured response with metadata.

Run:  python primary_agent.py
"""

import ollama
import json
import hashlib
import datetime


def query_primary_agent(question: str, model: str = "mistral") -> dict:
    """
    Sends a question to the primary LLM agent and returns a structured response.

    Args:
        question: The input question/prompt to evaluate.
        model:    The Ollama model to use (default: mistral).

    Returns:
        A dictionary containing the question, answer, model used, timestamp,
        and a hash of the response (used later for integrity checking).
    """

    system_prompt = (
        "You are a factual question-answering assistant. "
        "Answer the question as accurately and concisely as possible. "
        "Do not speculate. If you are unsure, say so."
    )

    print(f"\n[Primary Agent] Sending question to {model}...")
    print(f"  Question: {question}")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        options={"temperature": 0.5}  # explicit temp for reproducibility
    )

    answer = response["message"]["content"].strip()

    # Hash covers question AND answer so every block in the
    # blockchain audit log is unique even if answers are identical
    hash_input = f"{question}||{answer}"
    answer_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    result = {
        "agent":     "primary",   # used by cosine_layer.py
        "agent_id":  "primary",   # also used by cosine_layer.py
        "model":     model,
        "question":  question,
        "answer":    answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      answer_hash,
    }

    print(f"\n[Primary Agent] Answer received:")
    print(f"  {answer[:300]}{'...' if len(answer) > 300 else ''}")
    print(f"  Hash: {answer_hash[:16]}...")

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_question = "Who was the first person to walk on the moon, and in what year?"

    result = query_primary_agent(test_question)

    print("\n" + "="*60)
    print("FULL RESULT (JSON):")
    print("="*60)
    print(json.dumps(result, indent=2))