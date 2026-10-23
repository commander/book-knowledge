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
| E3 | 26.7% (40/150; direct 20, indirect 20) | +16 [9, 23], line 1 PASS | 3 / 56 / 1 | 4 / 21 / 5 | 0 | 1.91 h |
| E5 | 16.0% (24/150; direct 15, indirect 9) | +5 [−2, 13] | 0 / 60 / 0 | 4 / 21 / 5 | 0 | 2.34 h |
| E5a | 20.0% (30/150; direct 15, indirect 15) | +9 [1, 17], line 1 PASS | 1 / 58 / 1 | 5 / 17 / 8 | 0 | 1.4 h (train only, E5 data) |
| E5b | 22.0% (33/150; direct 15, indirect 18) | +11 [3, 20], line 1 PASS | 3 / 55 / 2 | 4 / 22 / 4 | 0 | 1.82 h |

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

## E3: data fixes, 3 question forms for each fact, passage rewrites (2026-10-22)

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

### E3 result

Pass lines: line 1 (knowledge) PASS, +16 points [9, 23]. Line 2 (restraint)
FAIL: traps accepted 56 (maximum 2), twins refused 5 (maximum 2). Line 3
(everyday) PASS. This is the first run that passes line 1.

Measured on a T4:

| Stage | Time | Notes |
|---|---|---|
| facts | 13.3 min | 2,645 facts (per unit median 7, maximum 12). Removed: not_sentence 27, no_quote 242, duplicate 16. Units with no JSON after the retry: 12 (E1: 69) |
| data | 32.0 min | fact_qa 5,559 rows (q1 2,024, q2 1,747, q3 1,788), 2,111 of 2,645 facts covered, 787 calls. Rewrites 694 rows (median 122 words) |
| train | 67.9 min | 6,253 rows, 465,650 tokens per epoch, 1,173 steps (3 epochs), loss 2.85 → 0.49. One CUDA OOM warning at step 8, recovered |
| save | 1.6 min | |
| total | 1.91 h | |

fact_qa pairs removed: no_json 1,185 (395 facts in calls with no JSON after the
retry), key_not_in_answer 859, forbidden_word 176, fact_missing 24, pronoun 21,
yes_no 12, duplicate 10, repeats_question 3, bad_form 1. The no_json loss is
the largest: 534 facts have no pairs.

Recall of the training questions (300 rows): fact questions 96 to 97%.
Rewrites 0%, because the check needed the full text word for word. The check
now counts a rewrite as recalled when half of its content words are in the
answer. The sample answers show that the model learned the start and the
subject of each rewrite, but not the full text.

Coverage of the dev knowledge questions (rough word match):

| | in unit data | elsewhere in data | not in data | unit has no data |
|---|---|---|---|---|
| direct (90) | 75 (18 correct) | 1 (0) | 14 (2) | 0 |
| indirect (60) | 53 (19 correct) | 2 (0) | 5 (1) | 0 |

E2 direct: in unit data 36 (7 correct), unit has no data 12.

Conclusions:
- The data now covers most exam answers (75 of 90 direct). The model answers
  about a quarter of these correctly. Many answers in the "in unit data" group
  are only in a rewrite (for example "New Zealand", "dungeonmaster"), not in a
  fact question. A fact that is only one word in a long text is not learned.
- The rewrites and the 3 question forms gave the gain. Indirect questions went
  from 10 to 20 correct without joined questions (E4).
- Traps: the model accepts almost all false premises (56 of 60, baseline 59).
  The training data has no example of a rejected premise. This is now the only
  failed line.
- 3 epochs at about 465k tokens per epoch did not memorise the data (end loss
  0.49, E2: 0.001). E3 used less GPU time than E2.


### E3 data check (`diagnose_run.py data`)

- facts: 83 replies fail the plain JSON parse, and 12 still fail after the
  quote repair. Causes in the examples: two objects joined into one (no
  `}, {` between them), and a quoted title after a comma
  (`track, "Sense of Doubt".`) that the quote repair did not escape.
- fact_qa: 111 of 787 replies have no JSON after the repair (395 facts lost).
  In all 5 examples, the teacher left out a closing bracket: `"}]]` (the
  object has no `}`) or `"}]}` (no outer `]`).

## E5: JSON bracket repairs, trap/twin pairs (2026-10-22)

Decision (2026-10-22): E5 comes before E4, because line 2 is the only failed
line. The JSON repairs go in the same run, to save Colab time. Part of the
knowledge change in E5 comes from the repairs (about 400 more facts with
questions).

Changes from E3:
- JSON repairs (`facts.py`, used by all teacher parts):
  - after a comma, a double quote opens a string only when a key follows
    (`"word":`);
  - a `"fact":` key after a comma gets the missing `}, {` before it;
  - missing closing brackets are added (`balance_brackets`);
  - an object with the same key two times is not valid (two joined objects).
- Trap/twin pairs (3d, `data.traps`, prompt `traps.txt`): for half of the
  facts (random, fixed seed), a true question (twin) and the same question
  with one false detail (trap). The twin answer states the fact with the key.
  The trap answer says that the false detail is wrong and gives the true one.
  No "I do not know" answers.
- Checks on each pair (pair removed): the questions differ by more than 4
  words; the false detail is not in the trap question or is also in the twin
  question; an added doubt word in the trap question; a twin answer without
  the key or with correction words; a trap answer without the true detail,
  without a negation, or with a "do not know" form; forbidden words, yes/no
  questions, pronouns, duplicates.
- Cue score (logistic regression on the words of the questions, 5 folds) is
  only logged in this run.
- Training: the same as E3.

Estimate: about 1,300 pairs (2,600 rows); teacher about 10 min more; training
about 90 min; total about 2.4 h.

What to look at: traps accepted and dodged, twins refused (risk: more refusals
of true questions), knowledge, the traps counts and cue score, the data check.
If twins refused goes up, E5b uses a pair for every third fact.

### E5 result

All lines except line 3 fail. Knowledge went down from 26.7% to 16.0%
(direct 20 → 15, indirect 20 → 9). Traps accepted 60 of 60.

Measured on a T4: facts 14.5 min, data 39.5 min, train 84.8 min (6,979 rows,
510,790 tokens per epoch, 1,309 steps, loss 2.55 → 0.20, 4 CUDA OOM warnings,
all recovered), save 1.7 min, total 2.34 h.

Data:
- The JSON repairs worked: replies with no JSON after the repairs: facts 10
  (E3: 12), fact_qa 6 of 787 (E3: 111), traps 14 of 457. fact_qa rows 6,076
  (E3: 5,559), facts with questions 2,348 (E3: 2,111). Some recovered fact_qa
  answers are only a phrase ("usually at night"), not a sentence.
- Traps: only 104 pairs of 1,322 chosen facts. The 7B teacher did not do the
  task: in 974 pairs the true and false questions are the same, and the false
  detail is only in the answer; 132 pairs are yes/no questions; others are
  statements, not questions. Many kept pairs are also poor ("It had a
  moralizing stance, not a letter, ...").
- Cue score 0.13 (no word cue, but on very few pairs).

Recall (300 rows): fact questions 84 to 89% (E3: 96 to 97%), twin 8 of 8,
trap 3 of 8.

Coverage: direct in unit data 75 (14 correct; E3: 18), indirect in unit data
57 (9 correct; E3: 19).

Example of a changed answer form: dev-116 (a normal knowledge question) got
"It is the collective boundary set by default retention periods, not
data-retention periods, that is encoded by software." This is the trap answer
form ("X, not Y") on a question with a true premise.

Conclusions:
- The trap/twin generation by the teacher in one step does not work with the
  7B teacher. The trap question must be made by code from a true question, so
  that the two questions are the same except for one detail.
- The cause of the knowledge loss is not known. Possible causes: the trap rows
  (the "X, not Y" form on true questions), the recovered fact_qa rows, or
  run-to-run variation. A train-only run on the E5 data without traps.jsonl
  separates the first cause from the others.


## E5a: E5 data without the trap rows (train only)

The E5 book folder copied to `work/state-e5a`, with `data/traps.jsonl` and the
train and save outputs removed. Only the train and save stages run. Purpose:
find if the trap rows caused the knowledge loss of E5.

### E5a result

- Training: 6,771 rows (E5 minus the 208 trap and twin rows), 1,270 steps,
  loss 2.85 -> 0.30, 78.7 min. The printed stage times and GPU total (3.70 h)
  include the E5 training, because the timing file came with the copied state.
- Knowledge 20.0% (direct 15, indirect 15), +9 [1, 17]: line 1 PASS, but
  below E3 (26.7%). Line 2 FAIL: twins refused 8 (limit 2).
- Coverage: direct with the answer in unit data 75 (15 correct; E3 18);
  indirect 57 (15 correct; E3 19).
- Recall of training questions: fact_q1 89%, fact_q2 95%, fact_q3 89%
  (E3 96-97%). Rewrites 0 of 26: the model does not repeat the long rewrite
  answers.
- Conclusion: the result is between E3 and E5. The trap rows removed about
  6 correct answers (E5 -> E5a), but the other 10 (E5a -> E3) come from the
  E5 data itself (new teacher samples, the recovered fact_qa rows) or from
  run-to-run variation. With 150 questions, one standard error is about
  3.5 points, so none of these differences is certain.
- Twins refused 8 with no trap rows in the data. So the twin refusals do not
  come from the trap rows.
- Decision: run E5b with `facts_share` 0.5 (no change).

## E5b: traps made by code

Changes from E5 (the trap part only):
- The twin is an existing fact question (`fact_q1` or `fact_q2`) that passed
  all checks. It stays in the data once, as a fact question.
- The teacher gets 8 questions in each call, each with its answer and fact,
  and gives only one `detail` (words copied from the question, not the
  answer) and a false `replacement` of the same type.
- The code replaces the detail in the twin question. So the trap and the twin
  are always the same question except for the detail.
- The trap answer is one of 5 opening forms (chosen with the seed) that names
  the false and the true detail, then the twin answer.
- A trap is removed when: the detail is not in the question, the detail is
  (part of) the key, the replacement is not a change, the replacement is in
  the unit text (it can be true), more than 4 words change, or the
  replacement has a forbidden word or a pronoun.
- Share: half of the facts that have a twin question. About 1,100 trap rows,
  against about 6,000 true fact questions.

The E5a result decides the amount: if the trap rows caused the loss in E5,
E5b starts with a smaller share.

### E5b result

- Data: facts 2,653; fact_qa 6,128 rows (2,367 facts); rewrites 700;
  traps 212 rows. JSON loss after the repairs: facts 9 units, fact_qa 5
  replies, traps 0.
- Traps: 1,168 facts chosen, 212 kept (18%). Dropped: missing item 364,
  detail not in the question 421 (the teacher took it from the answer or the
  fact), detail is the key 143, replacement in the unit 19, not changed 8,
  too many words 1. Cue score 0.196. Some kept traps are weak (general words:
  "toolkit" -> "manual", "now" -> "in the past").
- Training: 7,040 rows, 1,320 steps, loss 2.70 -> 0.37, 62 min.
- Knowledge 22.0% (direct 15, indirect 18), +11 [3, 20]: line 1 PASS.
  Traps 3 / 55 / 2, twins 4 / 22 / 4: line 2 FAIL.
- Recall: fact questions 93-95%, trap rows 7 of 13 (54%), rewrites 0 of 29.
- Coverage: direct with the answer in unit data 75 (14 correct); indirect
  57 (17 correct).

Conclusions:
- 212 trap rows do not change the exam trap result (55 accepted).
- E5, E5a and E5b give 16-22% with the same data method. E3 (26.7%) is
  about 1.5 standard errors above them. E3 can be a high sample.
- The model repeats 93-95% of its training answers, but it answers only
  about 19% of the exam questions whose answer is in the unit data. The
  problem is transfer from the training question forms to the exam forms,
  not memorization. Example: dev-116 has a training row with almost the same
  question and key, but the model gives a different answer.
- Line 2 needs at least 52 of 60 traps rejected. A model can reject a trap
  only when it knows the true detail. At 20-27% knowledge, line 2 cannot
  pass. Knowledge is the first problem.

