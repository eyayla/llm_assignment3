"""
Compute ROUGE and BERTScore metrics for all 3 checkpoints
against reference Alpaca answers.
"""

import json
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score

EVAL_DIR = "./eval_outputs"
OUTPUT_FILE = "./metrics_summary.json"

CHECKPOINTS = [
    "checkpoint_0_base",
    "checkpoint_1_alpaca",
    "checkpoint_2_json",
]

def load_alpaca_results(ckpt_name):
    path = os.path.join(EVAL_DIR, f"{ckpt_name}_alpaca.json")
    with open(path) as f:
        return json.load(f)

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": round(sum(r1)/len(r1), 4),
        "rouge2": round(sum(r2)/len(r2), 4),
        "rougeL": round(sum(rL)/len(rL), 4),
    }

def compute_avg_length(responses):
    lengths = [len(r.split()) for r in responses]
    return round(sum(lengths)/len(lengths), 1)

def compute_task_completion(responses):
    # Simple heuristic: response longer than 5 words = completed
    completed = sum(1 for r in responses if len(r.strip().split()) > 5)
    return round(completed / len(responses), 4)

def main():
    all_metrics = {}

    for ckpt in CHECKPOINTS:
        print(f"\nProcessing: {ckpt}")
        results = load_alpaca_results(ckpt)

        predictions = [r["response"] for r in results]
        references  = [r["reference"] for r in results]

        # ROUGE
        rouge = compute_rouge(predictions, references)
        print(f"  ROUGE: {rouge}")

        # BERTScore
        print("  Computing BERTScore (this takes ~1-2 min)...")
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        bertscore = round(F1.mean().item(), 4)
        print(f"  BERTScore F1: {bertscore}")

        # Output length
        avg_len = compute_avg_length(predictions)
        print(f"  Avg output length (words): {avg_len}")

        # Task completion
        completion = compute_task_completion(predictions)
        print(f"  Task completion rate: {completion:.1%}")

        all_metrics[ckpt] = {
            **rouge,
            "bertscore_f1": bertscore,
            "avg_output_length_words": avg_len,
            "task_completion_rate": completion,
        }

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {OUTPUT_FILE}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Checkpoint':<25} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERTScore':>10} {'Avg Len':>8} {'Completion':>10}")
    print("-" * 85)
    for ckpt, m in all_metrics.items():
        print(f"{ckpt:<25} {m['rouge1']:>8.4f} {m['rouge2']:>8.4f} {m['rougeL']:>8.4f} {m['bertscore_f1']:>10.4f} {m['avg_output_length_words']:>8.1f} {m['task_completion_rate']:>10.1%}")

if __name__ == "__main__":
    main()
