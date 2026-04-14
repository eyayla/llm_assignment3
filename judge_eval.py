"""
Judge Evaluation Script
Uses UTSA Llama 3.1 8B as judge to compare responses across checkpoints.
Evaluates pairwise: (ckpt0 vs ckpt1), (ckpt1 vs ckpt2), (ckpt0 vs ckpt2)
Scores each response on 6 dimensions.
"""

import os
import json
import time
import random
import requests
from pathlib import Path

# ---------------------------------------------------------
# UTSA API Configuration
# ---------------------------------------------------------
API_KEY  = "utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem"
BASE_URL = "http://149.165.173.247:8888/v1"
MODEL    = "meta-llama/Llama-3.1-8B-Instruct"

EVAL_DIR   = "./eval_outputs"
OUTPUT_DIR = "./judge_outputs"
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.5

# Pairs to compare
COMPARISON_PAIRS = [
    ("checkpoint_0_base", "checkpoint_1_alpaca"),
    ("checkpoint_1_alpaca", "checkpoint_2_json"),
    ("checkpoint_0_base", "checkpoint_2_json"),
]

# ---------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------
JUDGE_PROMPT = """You are an expert evaluator of language model responses. 
You will be given an instruction and two responses (Response A and Response B).
Evaluate both responses and return a JSON object with scores and a winner.

Instruction: {instruction}
{input_section}
Response A:
{response_a}

Response B:
{response_b}

Evaluate each response on these dimensions (score 1-5):
- instruction_following: Did the response follow the instruction?
- correctness: Is the response factually correct?
- clarity: Is the response clear and well-written?
- completeness: Is the response complete?
- structured_output_validity: Is any structured output (JSON, etc.) valid? (5 if not applicable)
- hallucination_risk: How likely is the response to contain hallucinations? (5=low risk, 1=high risk)

Return ONLY a JSON object in this exact format:
{{
  "response_a_scores": {{
    "instruction_following": <1-5>,
    "correctness": <1-5>,
    "clarity": <1-5>,
    "completeness": <1-5>,
    "structured_output_validity": <1-5>,
    "hallucination_risk": <1-5>
  }},
  "response_b_scores": {{
    "instruction_following": <1-5>,
    "correctness": <1-5>,
    "clarity": <1-5>,
    "completeness": <1-5>,
    "structured_output_validity": <1-5>,
    "hallucination_risk": <1-5>
  }},
  "winner": "<A or B or tie>",
  "justification": "<one sentence explanation>"
}}"""


def call_judge(instruction, input_text, response_a, response_b):
    input_section = f"Input: {input_text}\n" if input_text.strip() else ""
    
    # Randomly swap A/B to reduce position bias
    swapped = random.random() > 0.5
    if swapped:
        response_a, response_b = response_b, response_a

    prompt = JUDGE_PROMPT.format(
        instruction=instruction,
        input_section=input_section,
        response_a=response_a,
        response_b=response_b,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]).strip()
            
            result = json.loads(content)
            
            # If swapped, flip winner back
            if swapped:
                if result.get("winner") == "A":
                    result["winner"] = "B"
                elif result.get("winner") == "B":
                    result["winner"] = "A"
                result["response_a_scores"], result["response_b_scores"] = \
                    result["response_b_scores"], result["response_a_scores"]
            
            return result

        except Exception as e:
            print(f"  [Attempt {attempt+1}/{MAX_RETRIES}] Error: {e}")
            time.sleep(2)
    return None


def load_eval_results(ckpt_name, eval_type):
    path = os.path.join(EVAL_DIR, f"{ckpt_name}_{eval_type}.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        return json.load(f)


def compute_summary(results):
    if not results:
        return {}
    
    wins_a = sum(1 for r in results if r.get("winner") == "A")
    wins_b = sum(1 for r in results if r.get("winner") == "B")
    ties   = sum(1 for r in results if r.get("winner") == "tie")
    total  = len(results)

    dims = ["instruction_following", "correctness", "clarity",
            "completeness", "structured_output_validity", "hallucination_risk"]
    
    avg_a = {d: 0 for d in dims}
    avg_b = {d: 0 for d in dims}
    
    for r in results:
        for d in dims:
            avg_a[d] += r.get("response_a_scores", {}).get(d, 0)
            avg_b[d] += r.get("response_b_scores", {}).get(d, 0)
    
    for d in dims:
        avg_a[d] /= total
        avg_b[d] /= total

    return {
        "total": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / total,
        "win_rate_b": wins_b / total,
        "tie_rate": ties / total,
        "avg_scores_a": avg_a,
        "avg_scores_b": avg_b,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ckpt_a, ckpt_b in COMPARISON_PAIRS:
        for eval_type in ["alpaca", "json"]:
            print(f"\n{'='*60}")
            print(f"Judging: {ckpt_a} vs {ckpt_b} [{eval_type}]")
            print(f"{'='*60}")

            results_a = load_eval_results(ckpt_a, eval_type)
            results_b = load_eval_results(ckpt_b, eval_type)

            if not results_a or not results_b:
                print("  Skipping — missing eval results")
                continue

            # Match by prompt_id
            b_by_id = {r["prompt_id"]: r for r in results_b}
            paired = [(r, b_by_id[r["prompt_id"]]) for r in results_a if r["prompt_id"] in b_by_id]
            print(f"  Matched {len(paired)} pairs")

            judge_results = []
            for i, (ra, rb) in enumerate(paired):
                result = call_judge(
                    instruction=ra["instruction"],
                    input_text=ra.get("input", ""),
                    response_a=ra["response"],
                    response_b=rb["response"],
                )
                if result:
                    judge_results.append({
                        "prompt_id": ra["prompt_id"],
                        "checkpoint_a": ckpt_a,
                        "checkpoint_b": ckpt_b,
                        "eval_type": eval_type,
                        **result,
                    })
                
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(paired)} judged")
                time.sleep(SLEEP_BETWEEN)

            # Save results
            out_file = os.path.join(OUTPUT_DIR, f"{ckpt_a}_vs_{ckpt_b}_{eval_type}.json")
            with open(out_file, "w") as f:
                json.dump(judge_results, f, indent=2)

            # Print summary
            summary = compute_summary(judge_results)
            print(f"\n  Summary:")
            print(f"  {ckpt_a} wins: {summary['wins_a']} ({summary['win_rate_a']:.1%})")
            print(f"  {ckpt_b} wins: {summary['wins_b']} ({summary['win_rate_b']:.1%})")
            print(f"  Ties: {summary['ties']} ({summary['tie_rate']:.1%})")

            # Save summary
            summary_file = os.path.join(OUTPUT_DIR, f"{ckpt_a}_vs_{ckpt_b}_{eval_type}_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

    print(f"\nAll judge evaluations complete. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
