import ollama
import time
import pandas as pd

# ── Your models ─────────────────────────────────
MODELS = [
    "mistral",
    "llama3.2",
    "qwen2.5",
    "deepseek-r1",    # ← add this
]

# ── Prompt 1: The actual question ───────────────
PROMPT_1 = "who won the nobel price for mathametics in 2007"

# ────────────────────────────────────────────────
# ROUND 1: Get all model responses to Prompt 1
# ────────────────────────────────────────────────
print("=" * 60)
print("ROUND 1: All models answering the prompt")
print("=" * 60)

round1_results = {}

for model in MODELS:
    print(f"\n--- {model} responding ---")
    try:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT_1}
            ],
            options={"temperature": 0.7}
        )
        elapsed = round(time.time() - start, 2)
        output = response['message']['content']
        round1_results[model] = output
        print(f"✅ Done in {elapsed}s")
        print(output)

    except Exception as e:
        round1_results[model] = f"ERROR: {e}"
        print(f"❌ Error: {e}")

# ────────────────────────────────────────────────
# BUILD PROMPT 2: Bundle all Round 1 outputs
# ────────────────────────────────────────────────
bundled_responses = ""
for model, response in round1_results.items():
    bundled_responses += f"\n\n### Response by {model}:\n{response}"

PROMPT_2 = f"""You are a strict and impartial AI evaluation judge.

The following question was posed to multiple AI models:
QUESTION: "{PROMPT_1}"

Here are their responses:
{bundled_responses}

---
Evaluate each response on the following criteria:

1. FACTUAL ACCURACY — Are the facts stated correct?
2. HALLUCINATION — Did the model make up or fabricate anything?
3. COMPLETENESS — Did it fully answer the question?
4. REASONING QUALITY — Was the logic sound and well structured?

For each model provide:
- A hallucination score out of 10 (0 = none, 10 = severe)
- A quality score out of 10 (0 = poor, 10 = excellent)
- One specific example of what was wrong or hallucinated (if any)
- A final verdict: TRUSTWORTHY / PARTIALLY TRUSTWORTHY / NOT TRUSTWORTHY

End with a final summary stating which model gave the most reliable answer
and which gave the most hallucinated answer.
"""

# ────────────────────────────────────────────────
# ROUND 2: Each model evaluates all Round 1 outputs
# ────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("ROUND 2: Each model judging all responses for hallucination")
print("=" * 60)

round2_results = {}

for model in MODELS:
    print(f"\n--- {model} evaluating ---")
    try:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict and impartial AI evaluation judge. Be objective, precise and structured in your evaluation."},
                {"role": "user", "content": PROMPT_2}
            ],
            options={"temperature": 0.3}
        )
        elapsed = round(time.time() - start, 2)
        evaluation = response['message']['content']
        round2_results[model] = evaluation
        print(f"✅ Evaluation done in {elapsed}s")
        print(evaluation)

    except Exception as e:
        round2_results[model] = f"ERROR: {e}"
        print(f"❌ Error: {e}")
import re
import pandas as pd

# ── Step 1: Extract scores from Round 2 evaluations ──
def extract_scores(evaluation_text):
    """Pull hallucination and quality scores from evaluation text"""
    hallucination_scores = {}
    quality_scores = {}

    # Find all score patterns like "8/10" or "8 out of 10"
    lines = evaluation_text.split('\n')
    current_model = None

    for line in lines:
        # Detect which model is being discussed
        for model_name in round1_results.keys():
            if model_name.lower() in line.lower():
                current_model = model_name

        if current_model:
            # Extract hallucination score
            hall_match = re.search(r'hallucination[^\d]*(\d+)\s*(?:/|out of)\s*10', line, re.IGNORECASE)
            if hall_match:
                hallucination_scores[current_model] = int(hall_match.group(1))

            # Extract quality score
            qual_match = re.search(r'quality[^\d]*(\d+)\s*(?:/|out of)\s*10', line, re.IGNORECASE)
            if qual_match:
                quality_scores[current_model] = int(qual_match.group(1))

    return hallucination_scores, quality_scores

# ── Step 2: Aggregate scores across all judges ──
all_hallucination = {model: [] for model in round1_results.keys()}
all_quality = {model: [] for model in round1_results.keys()}

for judge, evaluation in round2_results.items():
    h_scores, q_scores = extract_scores(evaluation)

    for model, score in h_scores.items():
        all_hallucination[model].append(score)
    for model, score in q_scores.items():
        all_quality[model].append(score)

# ── Step 3: Build leaderboard ────────────────────
leaderboard = []
for model in round1_results.keys():
    h_scores = all_hallucination[model]
    q_scores = all_quality[model]

    avg_hallucination = round(sum(h_scores) / len(h_scores), 2) if h_scores else "N/A"
    avg_quality = round(sum(q_scores) / len(q_scores), 2) if q_scores else "N/A"

    leaderboard.append({
        "Model": model,
        "Avg Hallucination Score (lower=better)": avg_hallucination,
        "Avg Quality Score (higher=better)": avg_quality,
        "Judges Count": len(h_scores)
    })

# Sort by hallucination score (lower is better)
df_leaderboard = pd.DataFrame(leaderboard)
df_leaderboard = df_leaderboard.sort_values("Avg Hallucination Score (lower=better)")

print("\n🏆 FINAL LEADERBOARD\n")
print(df_leaderboard)
print(df_leaderboard.to_string())