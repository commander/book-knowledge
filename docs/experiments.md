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
| E1 | 9.3% (14/150; direct 5, indirect 9) | −1 [−7, 5] | 0 / 57 / 3 | 4 / 26 / 0 | 0 | 1.03 h |

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

## E1: facts and 3 questions for each fact (2026-10-21)

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

Result: no gain. Knowledge is the same as the baseline (the interval contains
0). Direct questions went down (13 → 5), indirect went up (7 → 9); both changes
are in the noise range. No trap is rejected (expected: E1 has no trap training).

Measured on a T4:

| Stage | Time | Notes |
|---|---|---|
| facts | 14.3 min | 366 units in 8 min 26 s, then a retry of 96 units in 2 min 8 s. 2,542 facts. Removed: no_quote 266, duplicate 5. **69 units (19%) have no facts** (no JSON after the retry) |
| data | 18.6 min | 734 calls in 15 min 30 s, about 380 output tokens/s. 5,072 pairs (direct 1,912, reworded 1,980, reverse 1,180), 2,292 facts covered. Removed: forbidden_word 1,221, key_not_in_question 989, key_not_in_answer 173, other 171 |
| train | 27.7 min | 951 steps (3 epochs of 317), 1.64 s per step; 297,838 tokens per epoch; about 575 tokens/s (E0: 425); loss 3.32 → 1.62. One CUDA out-of-memory warning at step 149; the allocator recovered |
| save | 1.3 min | |
| exam answers | 2.8 min | |

Open questions (diagnosis with `src/diagnose_run.py`):
- Why do 69 units give no JSON? (reply cut by `max_tokens`, or a different form)
- Which words does the forbidden-word check find? 1,221 removed pairs is too many
  for references to the source only.
- Are the exam answers in the training data (a data problem), or are they in the
  data and the model did not learn them (a training problem)?
- Does the model answer its own training questions? A final loss of 1.62 is high.
