# Sequential Instruction Tuning of a Small LLM: Forgetting Analysis & Judge Evaluation

**LLM & Agentic Systems — Assignment 3**  
**Student:** elif ercek | **UTSA Graduate Course**

---

## 1. Methodology

### Student Model Selection
I selected **Llama 3.2 3B** as the student model. This model is publicly available on HuggingFace, fits comfortably within UTSA HPC's V100 32GB GPU memory, and represents a competitive small-model baseline. While Phi-3.5 Mini was recommended, Llama 3.2 3B was chosen for its wide community adoption and well-studied fine-tuning behavior.

### Stage 1 Data: Alpaca Instruction Dataset
The first training stage used the **Stanford Alpaca dataset** (`tatsu-lab/alpaca`) containing 52,002 general instruction-following examples. After filtering 28 samples with empty outputs, 51,974 examples remained. A 90/10 train/validation split yielded 46,776 training and 5,198 validation examples. The dataset covers diverse tasks including open-ended generation, rewriting, brainstorming, summarization, and QA.

### Stage 2 Data: Teacher-Generated JSON Instruct Dataset
I constructed a 150-example JSON instruction dataset through **imitation learning** from Llama 3.1 8B Instruct (UTSA API). This is not classical knowledge distillation — the student only sees the teacher's final text outputs and trains with standard cross-entropy loss. The pipeline:

1. Designed diverse prompts covering 5 required task types
2. Fed each prompt to the teacher model (Llama 3.1 8B Instruct via UTSA API)
3. Validated every response for JSON correctness — invalid responses were discarded and regenerated
4. Paired validated responses with original prompts into (instruction, input, output) schema

**Dataset composition:**
| Task Type | Valid Examples | Discarded |
|-----------|---------------|-----------|
| JSON Extraction | 30 | 0 |
| Schema-Constrained Generation | 30 | 7 |
| Exact-Label Classification | 30 | 0 |
| JSON Repair | 30 | 0 |
| Tool-Call Argument Generation | 30 | 17 |
| **Total** | **150** | **24** |

### Fine-Tuning Pipeline
Both stages used **LoRA** (not 4-bit QLoRA due to hardware compatibility) on UTSA HPC V100 GPUs via SLURM batch jobs.

**Hyperparameters:**
| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Base model | Llama 3.2 3B | Llama 3.2 3B + Stage 1 adapter |
| LoRA rank | 32 | 32 |
| LoRA alpha | 64 | 64 |
| LoRA dropout | 0.05 | 0.05 |
| Learning rate | 2e-5 | 2e-5 |
| Max steps | 500 | 200 |
| Precision | BF16 | BF16 |
| Batch size | 4 | 4 |
| Gradient accumulation | 8 | 8 |

### Judge Model & Evaluation Protocol
The **judge model** was Llama 3.1 8B Instruct via UTSA API. Pairwise comparisons were made between all checkpoint pairs: (0 vs 1), (1 vs 2), (0 vs 2). To reduce position bias, response order was randomly swapped 50% of the time. The judge scored each response on 6 dimensions (1–5 scale): instruction following, correctness, clarity, completeness, structured output validity, and hallucination risk.

---

## 2. Experiments

### 2.1 Three-Checkpoint Comparison

| Checkpoint | JSON Validity | Alpaca Judge Win Rate | Avg Instruction Following | Avg Correctness |
|------------|--------------|----------------------|--------------------------|-----------------|
| Checkpoint 0 (Base) | 24% | 45% (vs Ckpt1) | 4.25 | 4.13 |
| Checkpoint 1 (Alpaca) | 100% | 41% (vs Ckpt0) / 26% (vs Ckpt2) | 4.27–4.40 | 4.17–4.26 |
| Checkpoint 2 (JSON) | 100% | 60% (vs Ckpt1) | 4.61 | 4.50 |

**Key finding:** Checkpoint 2 did NOT exhibit catastrophic forgetting. Instead, it outperformed Checkpoint 1 on Alpaca tasks (60% win rate vs 26%), while maintaining 100% JSON validity.

### 2.1b Automatic Metrics (ROUGE & BERTScore)

| Checkpoint | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Avg Length | Completion |
|------------|---------|---------|---------|--------------|------------|------------|
| Checkpoint 0 (Base) | 0.273 | 0.118 | 0.217 | 0.856 | 125.0 | 97% |
| Checkpoint 1 (Alpaca) | 0.430 | 0.219 | 0.353 | 0.887 | 79.8 | 82% |
| Checkpoint 2 (JSON) | 0.376 | 0.161 | 0.292 | 0.877 | 67.6 | 89% |

ROUGE and BERTScore show a slight drop from Checkpoint 1 to Checkpoint 2, even though judge scores improved. This suggests the JSON-tuned model produces more concise responses (avg length dropped from 79.8 to 67.6 words) that diverge from reference answers in wording but are judged as higher quality.

### 2.2 Alpaca Evaluation Results

**Checkpoint 0 vs Checkpoint 1 (Alpaca fine-tuning effect):**
| Metric | Ckpt 0 (Base) | Ckpt 1 (Alpaca) |
|--------|--------------|-----------------|
| Judge Win Rate | 45% | 41% |
| Tie Rate | 14% | 14% |
| Avg Instruction Following | 4.25 | 4.40 |
| Avg Correctness | 4.13 | 4.26 |
| Avg Clarity | 4.27 | 4.27 |
| Avg Completeness | 3.98 | 4.11 |
| Avg Hallucination Risk | 3.59 | 3.69 |

Stage 1 Alpaca fine-tuning showed modest improvements across all dimensions, though the base model was already competitive (45% win rate), suggesting Llama 3.2 3B has strong pre-trained instruction-following capabilities.

**Checkpoint 1 vs Checkpoint 2 (Forgetting analysis):**
| Metric | Ckpt 1 (Alpaca) | Ckpt 2 (JSON) |
|--------|-----------------|----------------|
| Judge Win Rate | 26% | **60%** |
| Tie Rate | 14% | 14% |
| Avg Instruction Following | 4.27 | **4.61** |
| Avg Correctness | 4.17 | **4.50** |
| Avg Clarity | 4.17 | **4.62** |
| Avg Completeness | 3.96 | **4.39** |
| Avg Hallucination Risk | 3.59 | **3.84** |

**No catastrophic forgetting was observed.** Checkpoint 2 outperformed Checkpoint 1 on all Alpaca dimensions.

### 2.3 JSON Structured Output Evaluation

| Checkpoint | JSON Validity | Judge Win Rate (JSON tasks) |
|------------|--------------|----------------------------|
| Checkpoint 0 (Base) | 24% | 18% (vs Ckpt1) |
| Checkpoint 1 (Alpaca) | 100% | 22% (vs Ckpt0) / 20% (vs Ckpt2) |
| Checkpoint 2 (JSON) | 100% | 24% (vs Ckpt1) |

Notably, even Checkpoint 1 (Alpaca-only fine-tuning) achieved 100% JSON validity. This suggests that general instruction tuning on Alpaca also improved structured output capability, possibly because the Alpaca dataset includes some formatting-heavy examples.

### 2.4 Forgetting Analysis

**Central finding: No catastrophic forgetting was observed.**

| Metric | Ckpt 1 → Ckpt 2 Change |
|--------|------------------------|
| Alpaca Judge Win Rate | +34% (26% → 60%) |
| Avg Instruction Following | +0.34 (4.27 → 4.61) |
| Avg Correctness | +0.33 (4.17 → 4.50) |
| Avg Clarity | +0.45 (4.17 → 4.62) |
| Avg Completeness | +0.43 (3.96 → 4.39) |
| JSON Validity | 0% change (100% → 100%) |

Stage 2 JSON fine-tuning did not degrade Alpaca capabilities — it actually improved them. Several factors likely prevented forgetting:

1. **Small Stage 2 dataset (150 examples):** Limited exposure reduced overwriting of Stage 1 knowledge
2. **Low learning rate (2e-5):** Conservative updates preserved existing weights
3. **Few training steps (200):** Insufficient to catastrophically overwrite Stage 1 representations
4. **LoRA architecture:** Low-rank adapters constrain the update space, limiting forgetting

### 2.5 Ablation Study

**Ablation: Effect of Stage 2 training steps on forgetting**

We trained Stage 2 for 200 steps (approximately 47 epochs on 135 training examples). The very high number of effective epochs (47x) on a small dataset could risk overfitting, yet no forgetting was observed. This suggests that for small Stage 2 datasets with LoRA, the forgetting risk is low regardless of epoch count, because the LoRA adapter capacity is the binding constraint — not the number of gradient steps.

A secondary observation: the Stage 2 eval loss reached 0.159 with 95% token accuracy, indicating strong JSON format memorization without degrading Alpaca performance.

---

## 3. Analysis

### Qualitative Comparison

**Example: Open-ended generation task**
- *Checkpoint 0 (Base):* Produces reasonable but sometimes incomplete responses, occasionally wandering off-topic
- *Checkpoint 1 (Alpaca):* More focused and structured, follows the instruction format reliably
- *Checkpoint 2 (JSON):* Maintains Checkpoint 1's instruction following while adding more precise and complete responses — the discipline learned from JSON formatting appears to transfer positively to general tasks

**Example: JSON extraction task**
- *Checkpoint 0:* Only 24% valid JSON — often produces prose descriptions instead of JSON objects
- *Checkpoint 1:* 100% valid JSON — surprising result suggesting Alpaca tuning improved formatting discipline broadly
- *Checkpoint 2:* 100% valid JSON — maintains perfect validity with slightly better schema adherence

### Failure Case Analysis

The main failure mode at Checkpoint 0 was **format non-compliance**: the base model frequently responded to JSON extraction prompts with natural language descriptions rather than JSON objects. After Stage 1, this was completely eliminated.

For Checkpoint 1 on JSON tasks, the judge scores were similar to Checkpoint 2 (20% vs 24% win rate, mostly ties at 56%), suggesting that for relatively simple JSON tasks in our evaluation set, both fine-tuned checkpoints performed comparably.

### Discussion: Why No Forgetting?

The absence of catastrophic forgetting is a meaningful finding. Several explanations:

1. **Dataset size asymmetry:** Stage 2 used only 135 training examples vs 46,776 in Stage 1. The small Stage 2 dataset cannot overwrite the large representational changes from Stage 1.

2. **LoRA constrains updates:** LoRA's low-rank decomposition limits the "volume" of weight space that can be modified per training run. This acts as an implicit regularizer against forgetting.

3. **Task complementarity:** JSON formatting and general instruction following may share underlying capabilities (structured thinking, following constraints), so Stage 2 training reinforces rather than conflicts with Stage 1 knowledge.

4. **Conservative hyperparameters:** The 2e-5 learning rate and 200-step budget were conservative enough to avoid aggressive overwriting.

This finding aligns with recent literature suggesting that sequential fine-tuning with LoRA is more resistant to catastrophic forgetting than full fine-tuning, particularly when the second-stage dataset is small.

---

## 4. Prompt Engineering

### Teacher Model Prompts (Imitation Learning)

The system prompt for teacher-model JSON generation emphasized strict JSON-only output:

```
You are a helpful assistant that always responds with valid JSON only.
Do not include any explanation, markdown, or extra text — output only the JSON object.
```

Initial iterations revealed that the teacher model (Llama 3.1 8B) frequently included markdown code fences (```json ... ```) around the JSON output. A post-processing step was added to strip these fences before JSON validation. The tool-call task type had the highest discard rate (17/47, 36%) because the model often produced inconsistent parameter naming (e.g., `params` vs `parameters` vs `args`), leading to schema mismatches.

### Judge Prompts

The judge prompt was designed to elicit structured JSON output with explicit scoring rubrics:

```
Evaluate each response on these dimensions (score 1-5):
- instruction_following, correctness, clarity, completeness,
  structured_output_validity, hallucination_risk
Return ONLY a JSON object...
```

Position bias was mitigated by randomly swapping Response A and B order in 50% of evaluations, then reversing the winner label if swapped.

---

## Appendix: Full Prompt Templates

### Alpaca Training Format
```
Below is an instruction that describes a task{input_block}.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:  (optional)
{input}

### Response:
{output}
```

### Teacher Generation System Prompt
```
You are a helpful assistant that always responds with valid JSON only.
Do not include any explanation, markdown, or extra text — output only the JSON object.
```

### Judge Evaluation Prompt
```
You are an expert evaluator of language model responses.
You will be given an instruction and two responses (Response A and Response B).
Evaluate both responses and return a JSON object with scores and a winner.

Instruction: {instruction}
Input: {input}  (if applicable)

Response A: {response_a}
Response B: {response_b}

Evaluate each response on these dimensions (score 1-5):
- instruction_following
- correctness
- clarity
- completeness
- structured_output_validity (5 if not applicable)
- hallucination_risk (5=low risk, 1=high risk)

Return ONLY a JSON object in this exact format:
{
  "response_a_scores": {...},
  "response_b_scores": {...},
  "winner": "<A or B or tie>",
  "justification": "<one sentence>"
}
```

### Example Judge Output
```json
{
  "prompt_id": "alpaca_042",
  "checkpoint_a": "checkpoint_1_alpaca",
  "checkpoint_b": "checkpoint_2_json",
  "response_a_scores": {
    "instruction_following": 4,
    "correctness": 4,
    "clarity": 4,
    "completeness": 4,
    "structured_output_validity": 5,
    "hallucination_risk": 4
  },
  "response_b_scores": {
    "instruction_following": 5,
    "correctness": 5,
    "clarity": 5,
    "completeness": 4,
    "structured_output_validity": 5,
    "hallucination_risk": 4
  },
  "winner": "B",
  "justification": "Response B was more complete and precise while maintaining the same instruction format."
}
```
