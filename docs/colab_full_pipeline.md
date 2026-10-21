# Colab guide: the full dummy project

This guide gives all phases of the dummy project in the order that you do them.
The design is in `book_knowledge_project.md`. The current state is in `agent.md`.

Many parts do not exist yet. Each step shows its state:

- **READY**: the script exists. You can do the step now.
- **MISSING**: the script does not exist. The step tells what it must do. Ask
  Claude to write it when you get to that step.

## 0. Status of the parts

| Part | Script | State |
|---|---|---|
| Book preparation (text and units) | `src/prepare_books.py`, `books.json` | READY |
| Question writing, checks, floor answers, exam selection | `src/exam_builder.py` | READY (`floor` on a GPU only, not tested on a GPU yet) |
| `floor` on the Mac (MLX or transformers) | `src/exam_builder.py` | MISSING (option 2, planned) |
| Exam config (seeds, settings, thresholds) | `src/make_exam_config.py` → `exam_config.json` | READY (see 1.3) |
| Exam runner: a model (base or trained) answers an exam | `src/lab_eval.py answer` | READY (not tested on a GPU yet) |
| Judge: labels the answers blindly with the API | `src/lab_eval.py judge`, `prompts/judge_*.txt` | READY (not tested with a real API yet) |
| Score report: four pass lines, paired bootstrap, retention | `src/lab_eval.py score`, `retention`, `hand-check` | READY |
| Baseline recipe: fixed teacher data + QLoRA training | `src/baseline_recipe.py teacher`, `train` | READY (not tested on a GPU yet) |
| Freelancer pipeline `run_book.py` | `src/run_book.py`, `src/book_pipeline/`, `pipeline_config.json`, `pipeline_prompts/` | E0 READY (stages split, data, train, save; not tested on a GPU yet). Other stages MISSING |
| Sealing script: selects and prepares the sealed books | | MISSING |
| Table parsing in PDFs | `src/pdf_to_text.py` | Temporary: rotated pages are skipped (`agent.md` 5c) |

## 1. Setup (one time)

### 1.1 Google Drive

Use the folder from `docs/colab_prepare_books.md` section 1:
`MyDrive/book-knowledge/` with `src/`, `prompts/`, `data/`, `books/`,
`books.json` and `requirements-prep.txt`. All output goes to `work/` on Drive,
so a disconnected Colab session loses nothing.

Planned folders in `work/`:

```
work/<book>/                  text and units (prepare_books.py)
work/familiarity/<book>/      familiarity check
work/exams/<book>/            exams (lab side)
work/runs/<run>/<book>/       model answers, labels and scores
work/state/                   weights and replay data between books (pipeline)
work/sealed/                  sealed books: never open
```

### 1.2 Runtimes

| Work | Runtime | Approximate cost |
|---|---|---|
| Book preparation, prompts, validate, select | CPU | none |
| `floor`, exam answers, teacher data, training | GPU: T4 (L4 when needed) | T4 about 1.8 compute units per hour, L4 about 4.8 |

Each time you change the runtime, mount Drive again and go to the folder:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/book-knowledge
```

On a GPU runtime, install vLLM before the first `floor`:

```python
!pip install -q vllm
```

### 1.3 Exam config (write it before the first exam)

The design requires that you write the seeds and the settings before the
first run, and that you do not change them after. Write `exam_config.json` in
the project folder, with:

- the seeds: `plan` seed and `select` seed for each book, and the bootstrap seed
- the answer settings: base model `Qwen/Qwen2.5-3B-Instruct`, default chat
  template, single turn, greedy decoding (temperature 0), 256 new tokens maximum
- the judge settings: the model name, temperature 0
- the pass lines: knowledge >= baseline + 8 points and bootstrap low end > 0;
  traps accepted <= 2, dodged <= 6; twins refused <= 2; everyday refused <= 2;
  retention >= 95%
- the bootstrap: 2,000 resamples

The scripts do not read this file yet. It is your record. The MISSING scripts
must read it.

To make the first version, run (one `--book` for each book whose exams exist):

```
!python src/make_exam_config.py --book critical-software-studies:book1:<select seed> --writer openai:gpt-6.1-sol
```

- `--book NAME:ROLE:SELECT_SEED`. Roles: `book1`, `practice_a`, `practice_b`.
  `select` does not save its seed, so you give it here.
- The script reads the plan seeds, the file hashes, the question counts and the
  question-writer model (`generate_log.jsonl`). `--writer` is used only for a
  book without that log.
- It writes `FILL_IN` for the values that are in no file (judge model,
  bootstrap seed, familiarity judgment) and lists them. Edit the file and replace
  each one.
- It warns if an exam file changed after `select`. It does not replace an
  existing `exam_config.json`. To add a book later, edit the file.

## 2. Phase 1a: prepare the books (READY)

Follow `docs/colab_prepare_books.md`. Result: `work/<book>/units.jsonl` for
each book.

## 3. Phase 1b: familiarity check (READY, except the Mac `floor`)

Purpose: the base model must not know the book. Reject a book if more than 20%
of the judged questions are correct.

Do these steps for each book. The example uses `critical-software-studies`.

**CPU runtime:**

```
!mkdir -p work/familiarity/critical-software-studies
!cp work/critical-software-studies/units.jsonl work/familiarity/critical-software-studies/

!python src/exam_builder.py plan --dir work/familiarity/critical-software-studies --exams familiarity --n-direct 36 --n-pairs 0 --n-trap 0 --seed <seed>

!python src/exam_builder.py prompts --dir work/familiarity/critical-software-studies --title "Critical Software Studies" --templates prompts --direct-per-batch 12
```

Write the questions: paste each file from
`work/familiarity/<book>/prompts/` into a new Claude chat. Save each reply as
`work/familiarity/<book>/raw/<same name>.json`. Then:

```
!python src/exam_builder.py validate --dir work/familiarity/critical-software-studies --ngram 100
```

- `validate` checks the questions. It does not decide about the book.
- `--ngram 100` switches off the copied-wording check. That check is necessary
  for the real exams, not for the familiarity check.
- You need about 30 kept questions. Do not run `plan` again in the same folder:
  the saved replies would not match the new prompt files.

**GPU runtime** (all books in one session):

```
!python src/exam_builder.py floor --dir work/familiarity/critical-software-studies --model Qwen/Qwen2.5-3B-Instruct --dtype half
```

On the Mac, `floor` needs the MLX option (MISSING, option 2).

**Judge:** make a table from `floor_familiarity.jsonl` with three columns:
question, answer key (answer and accepted variants), model answer. Do not
include the model name. Ask Claude chat for one label for each row: `correct`
or `incorrect`. An answer is correct only if it states the key fact. Read some
labels yourself.

**Decide:** reject the book if more than 20% are correct (for example, 7 or more
of 32). Write down the seed, the number of questions, the count and the decision.

## 4. Phase 1c: choose the roles

| Role | Book | Exams |
|---|---|---|
| Book 1 (development) | The book with the most units (300 or more) | `dev` and `acceptance` |
| Practice A, Practice B | The other two open books | `acceptance` only |
| Sealed 2, Sealed 3 | Selected by the sealing script | `acceptance` only |

Use books of different genres. This tests if the recipe transfers.

## 5. Phase 1d: build the exams (READY, `floor` on GPU)

Use a new folder for each book, so that the familiarity questions are not
reused.

**CPU runtime**, book 1:

```
!mkdir -p work/exams/critical-software-studies
!cp work/critical-software-studies/units.jsonl work/exams/critical-software-studies/

!python src/exam_builder.py plan --dir work/exams/critical-software-studies --exams dev acceptance --seed <plan seed>

!python src/exam_builder.py prompts --dir work/exams/critical-software-studies --title "Critical Software Studies" --templates prompts
```

For a practice book, use `--exams acceptance`.

`plan` selects more items than the exam needs (120 direct units, 80 pairs and
80 trap units for each exam). Book 1 gets 2 x 22 = 44 prompt files. A practice
book gets 22.

**Write the questions.** Two ways:

- Claude chat (open books only): paste each prompt file into a new chat and
  save the reply as `raw/<same name>.json`. 44 files take some time.
- API: the `generate` command (see "Question writing with an API" below).

Use a non-Qwen model. Do not use the pipeline teacher.

**Question writing with an API.** `generate` works with three providers. A CPU
runtime is sufficient.

1. One time: in the Colab **Secrets** panel (key icon), add the key of each
   provider that you use, and give the notebook access: `ANTHROPIC_API_KEY`,
   `OPENAI_API_KEY`, `GEMINI_API_KEY`.
2. Each session:

   ```python
   from google.colab import userdata
   import os
   for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
       try:
           os.environ[k] = userdata.get(k)
       except Exception:
           pass                      # this key is not in Secrets
   !pip install -q -r requirements-api.txt
   ```

3. Run it:

   ```
   !python src/exam_builder.py generate --dir work/exams/<book> --provider openai --model gpt-6.1-sol
   !python src/exam_builder.py generate --dir work/exams/<book> --provider gemini --model gemini-3.8-flash
   !python src/exam_builder.py generate --dir work/exams/<book> --provider anthropic --model claude-haiku-4-5
   ```

| Option | Default | Notes |
|---|---|---|
| `--provider` | `anthropic` | `anthropic`, `openai` or `gemini` |
| `--temperature` | anthropic 0.3; openai and gemini: not sent | A number, or `none` to send no temperature. Some models reject any value that is not their default (for example `claude-sonnet-5-5`, `claude-opus-5-5`, and some reasoning models): use `none` for them. |
| `--max-tokens` | anthropic 8000; openai and gemini 16000 | The output limit, thinking included |

- `generate` skips the batches that already have a reply in `raw/`, so you can
  run it again after a failure.
- A reply that was cut off (token limit) or blocked is not saved. The command
  prints `NOT SAVED (<reason>)` and stops with an error at the end. Run it again;
  for a token limit, use a larger `--max-tokens`.
- Each call is written to `generate_log.jsonl` in the book's folder: provider,
  model, settings, tokens, and if the reply was complete. Keep this file. It is
  the record of which model wrote the exam.
- OpenAI calls use `store=False`, so OpenAI does not keep the prompts and replies.
- To compare providers, run each one in a separate copy of the folder (same
  `plan.json` and prompt files, empty `raw/`), then run `validate` in each copy
  and compare the reject counts. Use only one provider for the final exam of a book.
- Model IDs and prices change. Check the provider's model page before you start,
  and write down the exact model ID.

**Check the questions:**

```
!python src/exam_builder.py validate --dir work/exams/critical-software-studies
```

Use the default checks here (do not use `--ngram 100`). Read
`rejects_<exam>.jsonl` to see why questions were rejected.

**GPU runtime**: remove the questions that the base model already answers:

```
!python src/exam_builder.py floor --dir work/exams/critical-software-studies --model Qwen/Qwen2.5-3B-Instruct --dtype half
```

**CPU runtime**: build the final exams:

```
!python src/exam_builder.py select --dir work/exams/critical-software-studies --everyday data/everyday.jsonl --seed <select seed>
```

Result for each exam: `exam_<exam>.jsonl` (255 questions: 90 direct, 60
indirect, 60 traps, 30 twins, 15 everyday), `exam_<exam>.sha256` and
`cue_check_<exam>.json`.

- If `select` prints "not enough ...", you need more questions. Ask Claude how to
  add batches. Do not run `plan` again in the same folder.
- If a cue-check score is above 0.60, the traps can be found from their words.
  Look at the trap questions, rewrite the questions, and run `validate` and `select` again.
- Check that the mean word counts of traps and twins are about the same.
- After this step, do not change the exam files. The `.sha256` file shows a change.
- Do not show the exam files to the pipeline. Do not use them for training data.

## 6. Phase 1e: floor and baseline scores (READY, first GPU run not tested)

For each book, each exam is answered two times:

1. **floor:** the base model, no training. This is a sanity check.
2. **baseline:** the fixed baseline recipe (`book_knowledge_project.md` 4.1).
   Its knowledge score is the reference for pass line 1.

All settings come from `exam_config.json`. The scripts stop if a `FILL_IN`
value that they need is not replaced, or if an exam file does not match its
SHA-256 in the config.

Output of each run: `work/runs/<run>/<book>/` (`answers_<exam>.jsonl`,
`labels_<exam>.jsonl`, `score_<exam>.json`, and a `.meta.json` file for each
step with the model, settings, GPU and time).

### 6.1 GPU runtime (T4): answers and baseline training

Each session: mount Drive, `%cd /content/drive/MyDrive/book-knowledge`, then:

```
!pip install -q -r requirements-gpu.txt
```

If Colab asks for it, select **Runtime > Restart session**, and run the `%cd` again.

For each book (example: book 1, both exams):

```
# floor: the base model answers
!python src/lab_eval.py answer --book critical-software-studies --exam dev --run floor
!python src/lab_eval.py answer --book critical-software-studies --exam acceptance --run floor

# baseline: teacher data, then training, then answers with the adapter
!python src/baseline_recipe.py teacher --book critical-software-studies
!python src/baseline_recipe.py train   --book critical-software-studies
!python src/lab_eval.py answer --book critical-software-studies --exam dev --run baseline --adapter work/runs/baseline/critical-software-studies/adapter
!python src/lab_eval.py answer --book critical-software-studies --exam acceptance --run baseline --adapter work/runs/baseline/critical-software-studies/adapter
```

- Run each command in its own cell. `teacher` and `train` are separate
  processes, so the teacher (7B) and the student (3B) are never on the GPU
  together.
- `teacher` writes `teacher_pairs.jsonl`. Its output line shows the number of
  pairs and the number of units with fewer than 5 pairs. If many units have fewer
  pairs, look at `teacher_raw.jsonl` before you train.
- `train` saves a checkpoint every 50 steps. If Colab disconnects, run the same
  command again: it continues from the last checkpoint.
- `answer` does not replace an existing answers file (use `--force`; the old
  file is kept as a backup).
- Approximate time on a T4 for one book: teacher 10-20 minutes, training 20-40
  minutes, each `answer` 2-5 minutes. Model downloads come first.
- For a practice book, use only the exams that it has.

### 6.2 CPU runtime: judge and score

Each session: mount Drive, `%cd`, the API key cell (section "Question writing
with an API"), and `!pip install -q -r requirements-api.txt numpy`.

```
!python src/lab_eval.py judge --book critical-software-studies --exam dev --run floor
!python src/lab_eval.py judge --book critical-software-studies --exam dev --run baseline
!python src/lab_eval.py score --book critical-software-studies --exam dev --run floor
!python src/lab_eval.py score --book critical-software-studies --exam dev --run baseline
```

Do the same for `--exam acceptance`, and for each practice book.

- The judge model comes from `exam_config.json` (`judge.provider`,
  `judge.model`, `judge.temperature`). If the model rejects the temperature,
  add `--temperature none` and write this in the config.
- **Use the same judge model for all runs that you compare.** A different judge
  can label the same answer differently, and the paired comparison then measures
  the judge, not the model.
- `judge` sends 8 calls at the same time (`--workers`). If it stops (API error,
  no valid label), run it again: it judges only the missing questions. `score`
  refuses to run while a label is missing.
- The judge prompts are in `prompts/judge_*.txt`. Their hashes are stored in
  `labels_<exam>.meta.json`.

`score` prints the knowledge score, the trap, twin and everyday counts, and
pass lines 2 and 3. Pass line 1 needs a baseline:

```
!python src/lab_eval.py score --book <book> --exam dev --run <pipeline run> --baseline baseline
```

This prints the difference in points and the 95% paired bootstrap interval
(seed from `bootstrap.seed`). The pipeline runs come in Phase 2.

### 6.3 Hand check (after each acceptance run)

```
!python src/lab_eval.py hand-check --book critical-software-studies --exam acceptance --run baseline
```

It prints 20 random judged answers and saves them to `hand_check_<exam>.txt`.
For each one, decide if you agree with the label. If you disagree with more
than 2: change the judge prompt in `prompts/judge_*.txt`, then judge all runs
again with `--force` (the old labels are kept as a backup), so that all runs use
the same prompt.

For the book 1 acceptance exam, the hand check shows some acceptance
questions. Read only these 20.

### 6.4 What to record

For each book and exam: the floor and baseline knowledge scores, the trap,
twin and everyday counts, and the hand-check result. These are the "Floor" and
"Baseline" rows of the final report (`book_knowledge_project.md` section 10).
A good result: floor near 0-10% (the floor filter removed the questions that the
base model knows), baseline clearly higher.

## 7. Phase 2: develop the pipeline on book 1 (E0 READY)

Design, experiment list and start values: `docs/pipeline_design.md`.

- Use only the book 1 **dev** exam. Do not run `hand-check` on the book 1
  acceptance exam during development.
- Use a **new state folder for each experiment** (for example
  `work/state-e0`). A state folder holds one chain of books. A merged 3B model
  uses about 6 GB on Drive: delete the state folders of old experiments.
- After each experiment, record together: knowledge score, trap labels, twin
  refusals, everyday refusals, and the time log.
- Cap for each book: 5 T4-hours, 10 hours wall clock, all stages and the exam
  answers included.

### 7.1 E0: the baseline data through the pipeline code

E0 checks that the pipeline code works from start to end, and measures the
speed on a T4. `pipeline_config.json` has the baseline teacher prompt and the
baseline training values, so the dev score must be near the baseline score
(10.7%). The paired interval against `baseline` must contain 0.

GPU runtime (T4):

```
!pip install -q -r requirements-gpu.txt
!python src/run_book.py --book work/critical-software-studies/book.txt --state work/state-e0
!python src/lab_eval.py answer --book critical-software-studies --exam dev --run pipeline-e0 --model work/state-e0/model
```

- `run_book.py` runs the stages split, data (teacher), train, save. Each stage
  is a separate process. If Colab disconnects, run the same command again: it
  skips the completed stages and the training continues from its last
  checkpoint.
- It prints only counts and times. The time log is in
  `work/state-e0/<book_id>/time_log.json`. It counts all sessions, also the
  sessions cut by a disconnect.
- It refuses lab-side paths (exams, `work/runs/`, `exam_config.json`).
- If you change `pipeline_config.json` after a book started, it refuses to
  continue. Use a new state folder.

CPU runtime (judge and score):

```
!python src/lab_eval.py judge --book critical-software-studies --exam dev --run pipeline-e0
!python src/lab_eval.py score --book critical-software-studies --exam dev --run pipeline-e0 --baseline baseline
```

Send me: the `run_book.py` output (counts and time by stage), the `answer`
time, and the `score` output. The times give the data amounts for E1.

### 7.2 Each next experiment (E1, E2, ...)

`pipeline_config.json` holds the settings of the current experiment (the
`experiment` field). Results and plans: `docs/experiments.md`.

GPU runtime:

```
!git pull origin main
!python src/run_book.py --book work/critical-software-studies/book.txt --state work/state-e1
!python src/lab_eval.py answer --book critical-software-studies --exam dev --run pipeline-e1 --model work/state-e1/model
```

CPU runtime (after the API key cell):

```
!python src/lab_eval.py judge --book critical-software-studies --exam dev --run pipeline-e1
!python src/lab_eval.py score --book critical-software-studies --exam dev --run pipeline-e1 --baseline baseline
```

Change `e1` to the number of the experiment. Delete the `work/state-eN`
folders of old experiments when their result is recorded (about 6 GB each).

## 8. Phase 3: open three-book test (MISSING, needs Phase 2)

From an empty `work/state/`: book 1, then Practice A, then Practice B, one
command each. Measure the four lines on each book. Use the book 1 dev exam for
retention after Practice B. Fix forgetting and time problems, then do it again.

## 9. Phase 4: freeze

Copy the pipeline to `pipeline/`. Record the SHA-256 hash of each file and pin all
package versions. After this, do not change any file.

## 10. Phase 5: acceptance run (MISSING: sealing script)

1. The sealing script (`book_knowledge_project.md` 2.4) selects 2 books from 8–10
   candidates with a seed that you write down first. It prepares them
   (`prepare_books.py` with `"sealed": true`), does their familiarity check and
   writes their exams with the API, and runs their floor and baseline. It prints
   only counts and scores. You select the candidates by metadata only.
2. From an empty `work/state/`: book 1, then Sealed 2, then Sealed 3, with the
   frozen pipeline.
3. Answer the acceptance exams, judge them, and apply the four lines on each
   book. Retention: after Sealed 3, on the book 1 acceptance exam.
4. Fill in the report table (`book_knowledge_project.md` section 10).

If you change any file during this run, the run fails.

## 11. Colab rules

- Save all output and checkpoints to Drive. Colab can disconnect at any time.
- Training must continue from the last checkpoint after a disconnect.
- The cap counts GPU time. Log the start and end time of each stage and the GPU
  name. Time lost to a disconnect in the acceptance run still counts.
- Keep the browser tab open for night runs.
