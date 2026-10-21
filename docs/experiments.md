# Experiment log (Phase 2, book 1 dev exam)

Book 1: critical-software-studies (366 units, 89,477 words). Exam: dev (150
knowledge, 60 traps, 30 twins, 15 everyday). Judge: OpenAI gpt-6.1-sol,
temperature null. Student: Qwen2.5-3B-Instruct. GPU: Tesla T4.

Design and experiment list: `docs/pipeline_design.md` section 8.

## Results

| Run | Knowledge | vs baseline (95% interval) | Traps rej / acc / dodged | Twins correct / wrong / refused | Everyday refused | GPU time |
|---|---|---|---|---|---|---|
| floor | 4.0% (6/150) | | 4 / 44 / 12 | 3 / 15 / 12 | 0 | |
| baseline | 10.7% (16/150; direct 10, indirect 6) | | 1 / 59 / 0 | 5 / 25 / 0 | 0 | |
| E0 | 13.3% (20/150; direct 13, indirect 7) | +3 [−3, 8] | 1 / 59 / 0 | 3 / 26 / 1 | 0 | 0.53 h |

## E0: baseline data through the pipeline code (2026-10-21)

Config: baseline teacher prompt (5 Q/A pairs per unit, temperature 0.7),
QLoRA rank 16, alpha 32, q/k/v/o, learning rate 2e-4, 3 epochs, batch 16.

Result: the same as the baseline (the interval contains 0). The pipeline code
is correct.

Measured on a T4:

| Stage | Time | Notes |
|---|---|---|
| data (teacher) | 7.6 min | 3 min load and compile; 366 units in 3 min 56 s, about 390 output tokens/s; 1,829 pairs |
| train | 19.0 min | 345 steps, 2.9 s per step; 140,917 tokens per epoch; about 425 tokens/s; loss 3.06 → 2.08 |
| save (merge) | 2.0 min | |
| exam answers | 3.0 min | 255 answers |

Problems found and fixed during E0:
- New transformers does not accept `warmup_ratio`: the pipeline gives `warmup_steps`.
- peft stops on the old torchao on Colab (0.10) when it merges: the save stage skips the torchao check.
- A judge model that rejects a temperature gives all labels `invalid`: set `"temperature": null` in `exam_config.json`. `judge` now stops after a failed test call and prints the error.

Conclusion for E1: the teacher is fast. The training speed (about 425 tokens/s,
with short examples) is the limit. Test a faster training setup before the
data grows.

## E1: facts and 3 questions for each fact (planned)

Decision (2026-10-21): all experiments use the 3B student, not 1.5B first.
E0 used only 0.53 h of the cap, and 3B results compare directly with the 3B
baseline and the pass lines.

Changes from E0:
- New stage `facts` (teacher, temperature 0.2): 6-12 facts for each unit, each
  with a key and a quote. Facts whose quote is not in the unit are removed.
- Data mode `fact_qa` (teacher, temperature 0.7): for each fact a direct, a
  reworded and a reverse question, with short, one-sentence and 1-2 sentence
  answers. Pairs are removed if they refer to the source ("the text", "this
  book"), are yes/no questions, repeat the question in the answer, or do not
  contain the key (direct answers, reverse questions).
- Training: the same values as E0 (rank 16, q/k/v/o, 3 epochs, lr 2e-4,
  batch 16), but 16 examples on the GPU at the same time (no accumulation)
  and length-grouped batches, for speed.

What to look at: knowledge (mainly direct), the number of facts and pairs,
the time of each stage, the training speed in tokens/s.
