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
| E6 | 30.0% (45/150; direct 24, indirect 21) | +19 [11, 28], line 1 PASS | 0 / 56 / 4 | 4 / 21 / 5 | 0 | 2.56 h |
| E6r | 26.0% (39/150; direct 20, indirect 19) | +15 [7, 23], line 1 PASS | 0 / 59 / 1 | 4 / 22 / 4 | 0 | 2.85 h (E6 repeat, seed 4321) |
| E7 | 26.7% (40/150; direct 25, indirect 15) | +16 [8, 24], line 1 PASS | 2 / 53 / 5 | 4 / 22 / 4 | 0 | 3.70 h |
| E8 | 32.7% (49/150; direct 27, indirect 22) | +22 [13, 31], line 1 PASS | 3 / 53 / 4 | 3 / 27 / 0 | 1 | 2.86 h |

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


## E6: transfer: 5 question forms for each fact, joined questions

Purpose: the model repeats 93-95% of its training answers but answers only
about 19% of the exam questions whose answer is in its data. E6 makes the
training questions more like the exam questions. Changes from E5b:
- fact_qa: 5 forms for each fact instead of 3 (3 facts in each call,
  2,048 tokens):
  1. "What/Which/Who" with the topic (as before);
  2. another start word, another side of the fact (as before);
  3. description -> name: "What term / name / concept ...";
  4. topic first: "In the history of ...", "In discussions of ...";
  5. the fact in new words. The code drops a form-5 question with 5 or more
     words in a row from the unit (`copies_passage`), the same limit as the
     exam check.
  The prompt has a full example of the 5 forms (Berners-Lee / CERN).
- joined (3b, new): for half of the facts (fact B), the fact A of another
  unit that shares the most rare words (in at most 2% of the facts). The
  teacher writes one question that starts from fact A and asks for the key
  of fact B ("Applying ...", "Unlike ...", "Which ... connects ..."), or
  skips the pair. Checks: the fact_qa checks, and the key of B must not be in
  the question. Rows: type `joined`, `fact_id` = B, `context_fact_id` = A.
- traps: off. At 20-27% knowledge, line 2 cannot pass (see E5b).
- Training: the same as E3-E5b.

Expected: about 10,000 fact rows, 1,000 joined rows, 700 rewrites. Training
about 100 min; total GPU about 3 h (cap 5 h).

### E6 result

- Data: facts 2,658 (9 units with no JSON); fact_qa 9,757 rows (q1 2,108,
  q2 1,893, q3 1,998, q4 1,932, q5 1,826; 2,205 facts); joined 464 rows;
  rewrites 695. `copies_passage` dropped 116 form-5 questions (about 6%).
  No reply was cut at 2,048 tokens. JSON loss after the repairs: facts 9
  units, fact_qa 9 replies (the teacher wrote several arrays one after the
  other), joined 2 replies.
- Joined: 1,327 pairs, 464 rows (35%). Dropped: key in the question 510,
  repeats the question 151, key not in the answer 117, skip only 41. Many
  pairs share only a common word ("overlapping", "before", "developers"),
  and some answers join the two facts into a false statement.
- Training: 10,916 rows, 2,047 steps, 96 min. Total GPU 2.56 h.
- Knowledge 30.0% (direct 24, indirect 21), +19 [11, 28]: line 1 PASS.
  Traps 0 / 56 / 4, twins 4 / 21 / 5: line 2 FAIL. Everyday refused 0.
- Recall: fact questions 94-100%, joined 17 of 18, rewrites 0 of 19.
- Coverage: direct with the answer in unit data 75 (22 correct, E5b 14);
  indirect 56 (21 correct, E5b 17).

Conclusions:
- The 5 question forms help transfer. Direct goes from 15 to 24, the largest
  change of all the runs. The method is the same as E5b except for the
  question forms and the joined rows, so the gain is not only a high sample.
- The indirect change (18 to 21) is in the noise. Only 464 joined rows were
  kept and many are weak, so this run does not show if joined helps.
- Many wrong answers name a related term or person from the same unit
  (Fuller for Chun, GPT-3 for Codex, "algorithmic auditing" for
  "triangulation"). The model knows the topic but not the exact key.
- Some keys are only in rewrite rows (dev-131), and the model does not
  repeat rewrites (recall 0%). The next coverage report shows how many
  in-data keys are only in rewrites.
- Unit U0065 (dev-177) has 0 facts because its facts reply had no valid
  JSON.


## E7: a second facts pass for the missed details

Purpose: in E6, an exam key with its own question rows is answered 43% of the
time. A key that is only in a rewrite row is answered 10% (direct) or 26%
(indirect) of the time. 50 knowledge items (33%) have their key only in
rewrites, and 14 more have no data. The fact extraction misses these details
(names, examples, numbers, quoted terms).

Coverage split (E6, `diagnose_run.py coverage`):

| Key | Direct | Indirect |
|---|---|---|
| in question rows | 44 (19 correct) | 37 (16 correct) |
| in rewrites only | 31 (3 correct) | 19 (5 correct) |
| not in data | 12 (2 correct) | 2 (0 correct) |

Changes from E6:
- facts pass 2 (`facts.more`, prompt `facts_more.txt`): the teacher gets the
  unit and the pass-1 facts and gives up to 8 facts that are not in the list
  (names, terms in quotes, numbers, examples, who did what, list items). The
  same checks as pass 1; a fact with a key or text of a pass-1 fact is a
  duplicate. A unit with no pass-1 facts gets "(none)" as the list, so
  pass 2 also gets back the units lost to JSON errors. Facts have a `pass`
  field (1 or 2). Pass-2 replies: `raw/facts_more.jsonl`.
- JSON repair: a reply with several values one after the other (E6:
  `]}]\n[{"fact": 2`, `]}{"fact": 2`, `]}>{"fact": 2`) is read value by
  value. An object with repeated keys that starts with "n" is split into
  two objects (E6 joined losses).
- joined: off. E6 does not show a gain, and some rows join two facts into
  a false statement.
- No change to the 5 question forms, the rewrites and the training.

Expected: about 4,400 facts, about 16,000 fact rows, 700 rewrites. Training
about 150 min; total GPU about 3.9 h (cap 5 h). If the facts count is much
higher, lower `facts.more.max_new`.

### E7 result

- Facts: pass 1 2,712, pass 2 1,473 new; total 4,185 (per unit min 1,
  median 12, max 18). No unit has 0 facts. JSON loss after the repairs is 0
  in facts, pass 2 and fact_qa.
- Some pass-2 keys are general phrases ("differences in approach", "failure
  of Cloudflare software"). Others are good details (ALGOL, grid-think).
- fact_qa 16,358 rows (3,751 facts), rewrites 696. Dropped: key not in the
  answer 2,477, forbidden word 916, fact missing 690, copies passage 202.
- Training: 17,054 rows, 3,198 steps, 146 min. Total GPU 3.70 h (E6 2.56 h).
- Knowledge 26.7% (direct 25, indirect 15), +16 [8, 24]: line 1 PASS.
  Traps 2 / 53 / 5, twins 4 / 22 / 4: line 2 FAIL. Everyday refused 0.
- Recall: fact questions 96-99%, rewrites 0 of 5.

Coverage split (E6 in brackets):

| Key | Direct | Indirect |
|---|---|---|
| in question rows | 56, 24 correct (44, 19) | 42, 13 correct (37, 16) |
| in rewrites only | 22, 1 correct (31, 3) | 14, 2 correct (19, 5) |
| not in data | 11, 0 correct (12, 2) | 2, 0 correct (2, 0) |

Conclusions:
- Pass 2 does its job on the data: 17 more exam keys now have question rows.
  But direct goes only from 24 to 25. A key with question rows is still
  answered about 43% of the time (direct), so more keys in the data does not
  give the same gain in answers.
- Indirect goes from 21 to 15. Joined rows were on in E6 and off in E7, so
  joined is the probable cause, but a change of 6 of 60 can also be noise.
  The run-to-run variation is not measured yet.
- Many wrong answers have a training row with the key, but the row asks
  from another direction. Example: the training question "What was the error
  in the learning management system?" has the key in the question, and the
  exam asks "Which misconfigured system ...?". fact_qa has no check for the
  key in the question (only joined has it).
- In indirect questions, the model often answers about the first topic of
  the question (the context unit) and not the key of the second unit, or it
  repeats the question.
- Cost: 45% more GPU time for 5 fewer correct answers. E7 as it is does not
  replace E6 as the best run.


## E6r: E6 repeat with another seed

Purpose: E5a to E7 differ by 3 to 9 correct answers. The run-to-run
variation is not known, so it is not clear which of these differences are
real. E6r runs the E6 settings again with seed 4321 (E6: 1234). The seed
changes the teacher samples and the training order.

- Config: `configs/e6r.json` (the E6 config of commit 2f28254, only the seed
  and the experiment name are different). Run with
  `--config configs/e6r.json --state work/state-e6r`.
- The code is the current code. Differences from the E6 code: the JSON
  repairs of E7 (E6 lost 9 units and 9 fact_qa replies to JSON errors), and
  the `key_in_question_check` option (off in this config).
- Expected: about 2.6 h GPU, as E6.

## E8: drop the fact questions that contain the key

Purpose: in E7, many wrong answers have a training row with the key, but the
training question contains the key ("What was the error in the learning
management system?", key: learning management system). Such a row teaches
the model to repeat the fact, not to find the key from a question. The
joined questions have this check since E6; the fact questions do not.

Changes from E6:
- fact_qa: `key_in_question_check`: a pair is dropped (`key_in_question`)
  if the question contains the key, or all words of 3 or more letters of the
  key, as full words (`key_named` in `data.py`). This check is stricter than
  the answer check: a short key ("AI") does not match inside a word.
- Prompts, facts (one pass), joined, rewrites and training: the same as E6.
  Seed 1234, as E6, so most teacher replies are the same as in E6.

Expected: fewer fact rows than E6 (9,757); the data log gives the
`key_in_question` count. Total GPU about 2.5 h or less.

### E8 result

- Facts: 2,710 (one pass). JSON loss after the repairs: 1 unit (U0051: the
  reply repeats the same "quote" until the token limit). E6 lost 9 units.
- fact_qa 9,127 rows (q1 2,093, q2 1,643, q3 1,838, q4 1,824, q5 1,729;
  2,327 facts). `key_in_question` dropped 1,594 pairs (about 12% of the
  pairs that the teacher gave). The other drops: key not in the answer
  1,669, forbidden word 472, fact missing 395. Joined 496 rows (516 pairs
  dropped for the key in the question), rewrites 699.
- Training: 10,322 rows, 1,936 steps, 110 min. Total GPU 2.86 h.
- Knowledge 32.7% (direct 27, indirect 22), +22 [13, 31]: line 1 PASS. This
  is the highest score so far. Traps 3 / 53 / 4, twins 3 / 27 / 0: line 2
  FAIL. Everyday refused 1 (line 3 PASS, limit 2).
- Recall: fact questions 96-100%, joined 15 of 15, rewrites 1 of 24.

Coverage split (E6 in brackets):

| Key | Direct | Indirect |
|---|---|---|
| in question rows | 48, 24 correct (44, 19) | 37, 18 correct (37, 16) |
| in rewrites only | 28, 3 correct (31, 3) | 20, 4 correct (19, 5) |
| not in data | 10, 0 correct (12, 2) | 1, 0 correct (2, 0) |

Conclusions:
- A key with question rows is now answered about 50% of the time (E6 43%).
  This agrees with the purpose of the check, but the gain over E6 is 4
  answers (45 to 49). The E6r result is necessary to know if this is more
  than the run-to-run variation.
- The check removes 1,594 rows and the run is shorter than E7 (2.86 h).
- A key that is only in a rewrite row is still almost never answered (7 of
  48). Examples: "the liberty machine", "la perruque", "the table", "the
  digital poorhouse". All are named terms in the unit that the facts pass
  did not take.

### E6r result

- Facts 2,726 (no unit lost); fact_qa 10,949 rows (2,484 facts); joined
  425 rows; rewrites 698. Training 12,072 rows (E6 10,916), 2,264 steps,
  111 min. Total GPU 2.85 h.
- Knowledge 26.0% (direct 20, indirect 19), +15 [7, 23]: line 1 PASS.
  Traps 0 / 59 / 1, twins 4 / 22 / 4: line 2 FAIL. Everyday refused 0.
- Recall: fact questions 97-100%, joined 10 of 12, rewrites 0 of 23.

Coverage split (E6 in brackets):

| Key | Direct | Indirect |
|---|---|---|
| in question rows | 45, 16 correct (44, 19) | 38, 16 correct (37, 16) |
| in rewrites only | 25, 3 correct (31, 3) | 14, 3 correct (19, 5) |
| not in data | 14, 0 correct (12, 2) | 5, 0 correct (2, 0) |

Conclusions:
- The same settings give 45 (E6) and 39 (E6r) correct answers: a
  difference of 6 answers (4 points) from the seed only. The binomial
  standard deviation of one run at 28% of 150 is about 5.5 answers
  (3.7 points), so this difference is normal.
- Consequence: a difference of 4-6 answers between two single runs does
  not show a real effect. E3 to E8 are all between 39 and 49 answers.
- E8 (49) is above both E6 runs (45, 39; mean 42). The difference to the
  mean is 7 answers, about one standard deviation of a difference. This
  supports the key check, but it does not prove it. A paired comparison on
  the same exam items (`lab_eval.py score --baseline pipeline-e6`) gives a
  smaller interval than two separate intervals against the baseline.
- The seed also changes the data: 1,156 more training rows in E6r than in
  E6 with the same settings.

### E8 against E6 and E6r (paired, same exam items)

`lab_eval.py score --run pipeline-e8 --baseline <run>`:

| E8 against | Difference | 95% interval |
|---|---|---|
| E6 (30.0%) | +3 points | [−4, 9] |
| E6r (26.0%) | +7 points | [−1, 15] |

Both intervals include 0, so E8 is not proven better than E6. E8 stays the
base for the next run: it is not worse, its data has no questions that
give the answer away, and it has fewer rows than E6r.

## E9: a terms pass for the named terms that have no fact (planned)

Purpose: in E8, 48 knowledge items have their key only in rewrite rows, and
the model answers 7 of them. Many of these keys are quoted terms or names in
the unit ("the liberty machine", "la perruque", "the digital poorhouse",
"Pieces of Eight"). E7 tried a full second facts pass: it added 1,473 facts
and 1.1 h, and many of the new keys were general phrases. E9 lets the code
choose the terms, so the teacher writes only a small number of facts.

Changes from E8:
- facts, terms pass (`facts.terms`, prompt `facts_terms.txt`), after pass 1:
  - The code finds candidates in each unit: quoted terms ("...", curly
    double quotes, curly single quotes) of 1 to 6 words, then runs of
    capitalised words of 2 or more words, then single capitalised words.
    A name must not be the first word of a sentence.
  - A candidate is dropped if a kept fact of the unit has it as its key or
    as a part of its key, or if it is in more than 2% of the units
    (`max_df`; a topic of the book, not a detail). At most 3 terms for each
    unit (`max_per_unit`), in the order above.
  - The teacher gets the unit and the numbered terms, and writes one fact
    for each term, or `skip`. The code sets the key to the term, drops a
    fact that does not contain the term (`term_not_in_fact`), and applies
    the pass-1 checks (quote in the unit, duplicates). The facts have
    `pass` 3. Replies: `raw/facts_terms.jsonl`.
- The new facts go through fact_qa (5 forms, key-in-question check) as all
  other facts.
- Seed 1234, as E8, so pass 1 and most fact_qa replies are the same as in
  E8. The comparison with E8 is paired.

Expected: a few hundred new facts, 1,500 to 2,500 more fact rows, about 30
more minutes; total GPU about 3.3 h. The data log gives the counts.

