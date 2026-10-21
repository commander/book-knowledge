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
| E2 | 15.3% (23/150; direct 13, indirect 10) | +5 [−2, 11] | 0 / 60 / 0 | 2 / 27 / 1 | 0 | 2.37 h (0.55 teacher from E1 + 1.82) |

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

### E1 diagnosis (`src/diagnose_run.py`)

Recall of 300 training questions (the model answers its own training data):
direct 28%, reworded 17%, reverse 79%, all 35%. **The model did not learn its
data.** This is the main cause.

Coverage of the 90 dev direct questions: the answer is in the training data of
its unit for 36 (3 correct), not in the data for 20, and the unit has no data
for 12 (the other 22 match only rows of other units, a weak match). About one
third of the direct answers are not in the data: this is the second cause.

Data losses:
- Facts with no JSON (69 units): all replies are complete arrays (finish
  `stop`). The teacher puts unescaped double quotes in the quote values
  (`"critical mass"`), so `json.loads` fails. Fix: structured output in vLLM,
  or a tolerant parser.
- Forbidden words: "passage" 934 + 333, "according to the" 326 (often a
  correct use such as "according to the Viable System Model"), "the author" 92.
  Fix: remove the source phrase ("in the passage") from the pair instead of
  dropping the pair, and match "according to the" only before passage, text,
  book, chapter or author.
- Reverse questions without the key: 1,122. The teacher does not follow this
  rule. The kept reverse pairs are easy (recall 79%) and do not help the exam.
- Fact quality: some facts are only a term ("fact": "critical mass", "key":
  "critical mass"), not a statement.

## E2: stronger training on the E1 data (2026-10-21)

Only the training changes: LoRA rank 64, alpha 128, all linear layers (q, k, v,
o, gate, up, down), 8 epochs, 8 examples on the GPU (2 steps of accumulation,
effective batch 16). The split, facts and data stages are copied from E1, so
the result shows only the effect of the training. The teacher time of E1
(32.9 min) is added to the E2 GPU time. When you copy a book folder to a new
state, remove `train/`, `done/train`, `done/save`, `merged.ok`, `time_log.json`
and `run.json` (the first E2 run kept `merged.ok`, and the save stage did not
merge).

What to look at: recall of the training questions (target: above 80%),
knowledge on dev, the training time.

Result: the best run so far, but line 1 fails (the interval contains 0).

Training: 2,536 steps (8 epochs of 317), 2.49 s per step, 107 min, about 370
tokens/s (rank 64 on all layers is slower than E1). Loss 2.56 → 0.001. The loss
falls in steps at each epoch start: the model memorises the exact rows. One
CUDA out-of-memory warning at step 224 (recovered).

Recall of 300 training questions: direct 100%, reworded 67%, reverse 71%. All
the "missed" examples repeat the trained answer word for word (the key is not in
the trained answer), so the real recall is near 100%. **The model now knows its
data. The training is no longer the limit.**

Coverage of the 90 dev direct questions: answer in the unit data 36 (7 correct,
E1: 3), not in the data 20 (1 correct), unit has no data 12 (2 correct).

Conclusion: the model answers the trained question form, but not a new form of
the same fact. Examples: the training row "What was he afraid of?" → "academic
plagiarism" (a question with no subject); "BlueSky" learned only as the answer
to "Which software uses a federated system approach?". Each fact needs more
and different question forms, and full statements of the fact in text form
(the passage rewrites of E3). The data bugs of E1 (69 units with no facts,
removed pairs) also limit the coverage.

## E3: data fixes, 3 question forms for each fact, passage rewrites (planned)

Decision (2026-10-21): the data fixes and the rewrites go in one run, to save
Colab time. The result shows the effect of all the changes together.

Changes from E2:
- Facts: a JSON reply with unescaped double quotes inside the values is
  repaired (`escape_inner_quotes`). The prompt asks for single quotes and for a
  full sentence for each fact. A fact with fewer than 5 words, or the same as
  its key, is removed (`not_sentence`).
- Questions (3a): 3 question forms for each fact, all asking for the key, each
  with a one-sentence answer that contains the key. No reverse questions.
  A source phrase ("in the passage", "according to the text") is removed from
  the pair, and the pair is kept. "according to the" is forbidden only before
  passage, text, book, chapter, excerpt or author. Questions with he, she, him,
  his or her are removed (`pronoun`).
- Passage rewrites (3c): 2 for each unit (an explanation, and a list of key
  points), each as the answer to a request that the teacher writes.
- Training: the E2 values, but 3 epochs (the data is about 2.5 times larger).

Estimate: teacher about 45 min, training about 2 h, total about 3 h.

What to look at: knowledge (direct and indirect), the facts and data counts,
recall, coverage, the time.
