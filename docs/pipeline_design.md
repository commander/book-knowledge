# Pipeline design (freelancer side)

Status: E0–E8 and E6r done (best: E8, 32.7%, key-in-question check; E6 repeat: 30.0% and 26.0%, so the run-to-run variation is about 4 points): stages split, facts, data (parts `baseline_qa`, `fact_qa`, `joined`, `rewrites`, `traps`), train (with length-grouped batches), save. The other stages do not exist yet. Results: `docs/experiments.md`.

This document gives the design of `run_book.py`, the pipeline that puts the
knowledge of one book into the weights of Qwen2.5-3B-Instruct. The rules and the
pass lines come from the project design (sections 1, 5 and 6). This document
does not change them.

## 1. Goal

For each book, one run of the pipeline must pass the four lines on the
acceptance exam:

| Line | Pass condition | Book 1 value now |
|---|---|---|
| 1. Knowledge | Score ≥ baseline + 8 points, and the low end of the 95% paired bootstrap interval > 0 | Baseline dev 10.7%, so the target is ≥ 19% (aim for 23–26%) |
| 2. Restraint | Traps: accepted ≤ 2 and dodged ≤ 6 (of 60). Twins: refused ≤ 2 (of 30) | Baseline: 59 accepted, 0 dodged, 0 twins refused |
| 3. Everyday | Refused ≤ 2 (of 15) | Baseline: 0 refused |
| 4. Retention | After the last book, ≥ 95% of the book 1 questions that were correct after book 1 are still correct | Not measured yet |

What the floor and baseline runs on book 1 show:

1. The baseline learns few facts (+7 points over the floor, interval [1, 12]).
   Indirect questions are the weakest part (6 of 60).
2. The baseline accepts almost all false premises (59 of 60). Q/A training
   teaches the model to answer every question.
3. Many baseline answers only repeat the question as a statement. The model
   learned the form of an answer, but not the fact.

The pipeline must correct all three.

## 2. Rules

1. **One command per book, no changes between books:**
   ```
   python src/run_book.py --book <book.txt> --state <state dir>
   ```
2. **No exam access.** The pipeline reads only: the book file, the `--state`
   folder, its own code, its own prompts and `pipeline_config.json`. It must
   not open `work/runs/`, exam files, candidate files or `exam_config.json`. The
   code makes a check of all input paths at the start.
3. **No lab code.** The pipeline does not import `exam_builder.py` or
   `lab_eval.py`. Code that both sides need (for example the split) is copied
   into the pipeline. Then a change on the lab side cannot change the pipeline.
4. **No person and no API during the run.** The local teacher
   (Qwen2.5-7B-Instruct-AWQ in vLLM) makes all the data.
5. **Time cap: 5 T4-hours and 10 hours wall clock for each book.** The cap
   includes all stages and the exam answers. A Colab disconnect does not stop
   the clock.
6. **Frozen settings.** All settings are in `pipeline_config.json`. After the
   freeze (Phase 4), the code, the prompts and this file do not change.

## 3. Inputs and outputs

Input: the clean book text (`work/<name>/book.txt`, from `prepare_books.py`).

`state/` after a run:

```
state/
  model/                     merged fp16 model after the last book (the next book starts here)
  replay/<book_id>.jsonl     a sample of the training data of each book
  general.jsonl              general instruction data (made once, on the first book)
  books.json                 the list of books done: id, SHA-256, order, time
  <book_id>/
    units.jsonl              stage 1
    facts.jsonl              stage 2
    data/*.jsonl             stage 3 (one file for each data type)
    checks.json              stage 4
    train/                   stage 5 checkpoints (removed after stage 7)
    probes.jsonl             stage 6 probes and results
    time_log.json            stage 8
    done/<stage>             a marker for each completed stage
```

`<book_id>` is the first 12 characters of the SHA-256 of the book text.

The lab then answers the exam with the merged model:

```
python src/lab_eval.py answer --book <name> --exam dev --run pipeline-v1 --model state/model
```

The answer time from `answers_<exam>.meta.json` is added to the time log for
the cap.

## 4. Stages

Each stage is a separate process. When a process stops, all its GPU memory is
released, so the teacher and the student are never on the T4 at the same time.
Each stage writes a `done/<stage>` marker. When `run_book.py` starts again after
a disconnect, it skips the completed stages, and the training stage continues
from its last checkpoint.

### Stage 1: split

- Split the book into units of 150–250 words at paragraph limits (the same
  method as the lab, but a copy of the code).
- Output: `units.jsonl` with `U0001`, `U0002`, ...

### Stage 2: extract facts (teacher)

- For each unit, the teacher writes a list of atomic facts as JSON. Each fact
  has: the fact sentence, the key term (name, number, date, concept, system,
  person), and the type.
- Each fact must contain a quote from the unit. A fact with no quote match is
  removed. This removes facts that the teacher invented.
- Target: 6–12 facts for each unit. Duplicate facts (same key term and the
  same meaning) are removed.

Reason: the baseline prompt "Write 5 question-answer pairs about this passage"
gets the easy facts only. A fact list covers more of each unit.

### Stage 3: generate training data (teacher)

| Type | What | Amount (first estimate) | For which line |
|---|---|---|---|
| 3a Direct questions | 3 different questions for each fact: direct, reworded, reverse ("What is X?" and "Which concept is X?") | 3 for each fact | 1 |
| 3b Joined questions | Questions that need facts from two units. The unit pairs share a key term or a person | 1–2 for each unit | 1 (indirect) |
| 3c Passage rewrites | Each unit rewritten in 2 different ways, as an answer to a request such as "Explain ... " | 2 for each unit | 1 |
| 3d Trap/twin pairs | From one fact: a false question (one detail changed: a name, a number, a system) and the true question in the same words. The target answer for the false question rejects the false detail and gives the true one. The true question gets a normal answer | 1 pair for each 2 facts | 2 |
| 3e General data | Everyday requests with good answers (see section 5) | 5–10% of the total | 3 |
| 3f Replay | A sample from `state/replay/` of all earlier books | 15–20% of the total | 4 |

Rules for all answers:

- **Many answer styles.** Short, one sentence, with context. Random selection
  with a fixed seed. No fixed opening words (no "According to ...", no "From
  section ...").
- **The answer gives the fact early.** The key term is in the first sentence.
  This works against the "repeat the question" problem.
- **No reference to "the book", "the text" or "the passage"** in a question or
  an answer. The exam questions do not have them.
- **Trap answers reject and correct.** Example: "No, the credit-system analogy
  is from Baudrillard, not Lyotard. In it, ..." A reply such as "I do not know"
  is a `dodged` label in the exam, so the training data does not have this
  form.
- **Twin answers have no doubt words.** The model must answer a true question
  normally.

All training examples use the default Qwen chat template with one user turn and
no system message, the same as the exam. The loss is on the answer tokens only.

### Stage 4: checks on our own data

The same automatic checks as the exam builder, applied to our training data:

- Trap and twin differ by at most 4 words, and the trap has no added doubt
  words.
- No yes/no questions. No reference to "the book" or "the text".
- Cue-word check: a logistic regression on the words of the trap and twin
  questions (5-fold). If the score is above 0.60, the traps can be found from
  the words alone, and the model learns the words and not the facts. The
  pipeline then removes the pairs with the strongest cue words and logs the
  count.
- Duplicate removal.

### Stage 5: train (student)

| Setting | Baseline | Pipeline (start value) |
|---|---|---|
| Method | QLoRA 4-bit nf4, fp16 | Same |
| Target layers | q, k, v, o | All linear layers (q, k, v, o, gate, up, down) |
| Rank / alpha | 16 / 32 | 64 / 128 (test 128) |
| Learning rate | 2e-4 | 1e-4 to 2e-4, cosine, 3% warmup |
| Epochs | 3 | 2–3 (selected by stage 6) |
| Batch | 16 | 16 |
| Max length | 512 | 512 (passage rewrites can need 768) |
| Start model | Base model for each book | `state/model` (base model for the first book) |

The training saves a checkpoint at the end of each epoch, and also at fixed
step intervals for a restart after a disconnect.

### Stage 6: self-check and checkpoint selection

The pipeline cannot use the exam. It uses its own probes:

- **Knowledge probes:** for 5% of the facts, one question variant is kept out
  of training (the fact itself stays in training through its other questions).
- **Trap probes:** 30 trap/twin pairs are kept out of training.
- **General probes:** 20 general requests are kept out of training.

Scoring is by rules, with no judge model:
- knowledge: the key term (or a variant) is in the answer;
- trap: the answer contains the true detail, and it is not an answer that uses
  the false detail;
- twin and general: the answer has no refusal words.

The pipeline answers the probes with each checkpoint (vLLM with the LoRA
adapter, greedy, 256 tokens) and keeps the checkpoint with the best combined
score. The weights of the combined score are in `pipeline_config.json`.

These rules are less exact than the lab judge. We compare them with the lab
judge in Phase 2. If they select the wrong checkpoint, we change them.

### Stage 7: save

- Merge the selected adapter into the fp16 weights of the start model, and
  save the result as `state/model`.
- Save a sample of this book's training data in `state/replay/<book_id>.jsonl`.
- Remove the checkpoints.
- Run the probes again on the merged model. The merge of an adapter trained on
  4-bit weights into fp16 weights can change the results a little. A large drop
  is logged as a warning.

### Stage 8: time log

`time_log.json` records the start time, the end time and the GPU name for each
stage, and the total. If the total of a completed stage goes above its budget
(section 6), the log shows a warning. If the total goes above the cap, the run
fails.

## 5. General data

The pipeline must not use `data/everyday.jsonl`, because that file is part of
the exam.

Plan: take 500 requests from a public instruction data set with a permissive
licence (for example `databricks-dolly-15k`, CC BY-SA 3.0). The download is
pinned by version and SHA-256. The teacher writes the answers. This is done
once, on the first book, and saved in `state/general.jsonl`.

## 6. Time budget on a T4 (estimate, to be measured)

| Stage | Budget |
|---|---|
| 1 Split | < 1 min |
| 2 Facts (teacher) | 30 min |
| 3 Data (teacher) | 60–75 min |
| 4 Checks | 2 min |
| 5 Train | 120 min |
| 6 Self-check | 15 min |
| 7 Save (merge, probes) | 15 min |
| Exam answers (lab) | 15 min |
| Reserve | 30–45 min |
| **Total** | **≤ 5 h** |

Rough data size for a book of 366 units: about 3,000 facts, 9,000 direct
questions, 500 joined questions, 700 passage rewrites, 1,500 trap/twin pairs,
plus general and replay data. That is about 15,000 examples and 2 million
tokens for each epoch. The first measurement in Phase 2 gives the real speed of
the teacher and the training on a T4. Then we set the amounts so that the total
stays below the cap.

## 7. Code layout

```
src/run_book.py                 entry point: runs the stages in order, checks the inputs and the cap
src/book_pipeline/
  split.py                      stage 1
  teacher.py                    vLLM teacher: batch generation, JSON parsing, retries
  facts.py                      stage 2
  data.py                       stage 3 (3a-3f) and the answer styles
  checks.py                     stage 4
  train.py                      stage 5
  selfcheck.py                  stage 6
  save.py                       stage 7
  timelog.py                    stage 8
pipeline_prompts/               teacher prompts (facts, questions, joined, rewrites, traps, general)
pipeline_config.json            all settings and seeds
```

## 8. Development plan (Phase 2)

Rules for development:

- Use only the **dev exam of book 1**. Do not read the book 1 acceptance exam,
  its answers or its labels during development. (Run `hand-check` with
  `--exam dev` only.)
- Change one part at a time. After each change, record all four lines
  together. A change that improves traps often makes twins worse.
- All experiments use Qwen2.5-3B (decision of 2026-10-21: E0 used only
  0.53 h of the cap, so 1.5B tests are not necessary).

Order of the experiments:

| # | Change | What we look at |
|---|---|---|
| E0 | Stages 1, 5, 7 with the baseline data (the code runs from start to end) | Same result as the baseline |
| E1 | Fact extraction + 3 questions for each fact (3a) | Knowledge, direct questions |
| E2 | All linear layers, rank 64 | Knowledge |
| E3 | Passage rewrites (3c) | Knowledge |
| E4 | Joined questions (3b) | Knowledge, indirect questions |
| E5 | Trap/twin pairs (3d) | Traps accepted and dodged, twins refused |
| E6 | General data (3e) | Everyday refused |
| E7 | Learning rate and epochs; self-check selection | All lines; compare the self-check with the lab judge |
| E8 | Full run on 3B, with the time log | All lines, time cap |

Phase 2 ends when the book 1 dev exam on 3B shows +12 to +15 points over the
baseline, lines 2 and 3 pass, and the time log is below the cap. Then Phase 3:
book 1 → Practice A → Practice B from an empty `state/`, with replay (3f), and
line 4 on the book 1 dev exam.

## 9. Risks

| Risk | Effect | Action |
|---|---|---|
| Traps against twins | Too much rejection training makes the model refuse true questions | Equal pairs in 3d; twin answers with no doubt words; watch twins refused in each experiment |
| Time cap | Less data or fewer epochs than needed | Measure in E0; set the data amounts from the measured speed |
| Teacher quality (7B, AWQ) | Wrong facts or bad traps in the training data | Quote check in stage 2; checks in stage 4; read samples of each data type in Phase 2 |
| Forgetting | Line 4 fails after 3 books | Replay (3f); test a lower learning rate for later books |
| Merge loss (QLoRA to fp16) | The merged model is worse than the adapter | Probes again after the merge (stage 7) |
| Overfitting to the dev exam | Good dev score, bad acceptance score | Do not look at the acceptance exam; Practice A and B check the transfer |
| Colab disconnect | Lost time | Stage markers and checkpoints; the clock still counts |

## 10. Decisions (start values)

These are the start values. An experiment can change a value only when it
shows a clear gain on the dev exam.

| # | Decision | Start value | Reason | Test in |
|---|---|---|---|---|
| 1 | Form of the passage rewrites | Chat answers: a user request ("Explain ...", "Tell me about ...") and the rewrite as the answer. Loss on the answer only | The same form as the exam. Plain text with the loss on all tokens can weaken the chat behaviour (line 3) | E3. Test plain text only if E3 gives less than +2 points |
| 2 | LoRA rank | 64, alpha 128 | Rank 128 uses more memory and time on a T4, and the gain for a 3B model is usually small. Rank 64 on all linear layers is already 4 times the baseline rank, on more layers | E2. Test 128 only if knowledge stops increasing |
| 3 | Self-check selection | Two limits, then one score. Limits: twins refused ≤ 2 of 30, general refused ≤ 2 of 20. Of the checkpoints in the limits, keep the one with the highest (knowledge % + traps rejected %) | The same form as the pass lines: lines 2 and 3 are limits, line 1 is a score. If no checkpoint is in the limits, keep the one with the fewest refusals | E7 |
| 4 | General data | `databricks-dolly-15k` (CC BY-SA 3.0), only rows with an empty `context` field, 500 rows, fixed seed. Answers by the teacher | Rows with no context are complete requests, like the everyday exam. Answers from the teacher keep the answer style the same as the book data | E6 |
| 5 | Replay | 20% of the training examples of the new book, with an equal share for each earlier book | Line 4 needs 95% retention, so start with the higher value. An equal share keeps book 1 in the data after many books | Phase 3. Reduce to 15% only if line 4 passes with a margin and the time is short |
