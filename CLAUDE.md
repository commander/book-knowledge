# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo is a rehearsal ("dummy project") for a book-knowledge fine-tuning task. The task: put the knowledge of a book into the weights of a small Qwen2.5 model (no RAG at answer time), with a frozen pipeline that must transfer to unseen books.

Read these files before you start work:

- `agent.md` (local only, not in the public repository): handover notes. It gives the current task, known bugs, decisions, and the next steps. It is the most recent source. If it disagrees with `book_knowledge_project.md`, `agent.md` is correct (for example, the books are open-access books published after 30 June 2024, not Project Gutenberg books).
- `book_knowledge_project.md` (local only, not in the public repository): full design. It gives the exam format, pass lines, baseline recipe, pipeline stages, and compute budget.

## Rule: sealed books

`books/sealed/` holds the acceptance books. Do not open, list the content of, stage, print or summarize anything in `books/sealed/`. Do not read exam, candidate, floor or reject files made from it. Scripts can process these files, but they must print only counts and scores. If this rule is broken, the user must get new sealed books.

## Two roles, kept separate

- **Lab side** (exists now): book cleaning (`src/pdf_to_text.py`), exam building (`src/exam_builder.py`), prompts in `prompts/`, the shared everyday set `data/everyday.jsonl`. Exams are written and judged by a non-Qwen model (Claude chat, or the Anthropic, OpenAI or Gemini API), never by the pipeline teacher. Use one question-writer model for each book.
- **Freelancer side** (not written yet): `run_book.py --book <txt> --state state/`. One command per book, no changes between books. `state/` carries weights and replay data from book to book. The pipeline must not see the exams.

## Environment

- Mac (arm64): use `src/.venv`, or `pip install pymupdf beautifulsoup4 lxml`. The system `python3` does not have these packages.
- `exam_builder.py generate --provider anthropic|openai|gemini` needs the SDKs in `requirements-api.txt` and `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`. It does not save a cut-off or blocked reply, and logs each call to `generate_log.jsonl`. `--temperature none` sends no temperature (needed for models that reject non-default values). `select` needs `scikit-learn`.
- `exam_builder.py floor` and all training run on Google Colab (T4/L4) with vLLM. T4 has no bf16: use fp16 / `--dtype half`. Do not load the teacher and the student on a T4 at the same time.
- There are no tests, linter or build step.

## Commands

Full guide for all phases, with the state of each part (READY / MISSING): `docs/colab_full_pipeline.md`. Phase 1e (lab side): `src/lab_eval.py` (`answer` vLLM GPU, `judge` blind API with `prompts/judge_*.txt`, `score` with paired bootstrap, `retention`, `hand-check`) and `src/baseline_recipe.py` (`teacher` vLLM AWQ, `train` QLoRA fp16); both read `exam_config.json` and write to `work/runs/<run>/<book>/`. GPU packages: `requirements-gpu.txt` (not pinned yet). The GPU commands are not tested on a GPU yet. Planned next code change: a Mac `floor` option (MLX or transformers) in `exam_builder.py`; the user will ask for it.

Book preparation pipeline (the user runs it in Colab, CPU runtime; guide: `docs/colab_prepare_books.md`). `books.json` lists the books and their converter options; `prepare_books.py` converts and splits every book into `work/<name>/` (`book.txt`, `book.report.txt`, `units.jsonl`, `convert.log`, `prepare.json` with hashes and versions). `"sealed": true` makes the reports counts-only. Package versions are pinned in `requirements-prep.txt`.

```
python src/prepare_books.py --config books.json [--only <name> ...]
```

The single steps (use the EPUB when it exists; use the PDF only when there is no EPUB):

```
src/.venv/bin/python src/epub_to_text.py list    books/<name>.epub
src/.venv/bin/python src/epub_to_text.py convert books/<name>.epub books/<name>.txt
src/.venv/bin/python src/pdf_to_text.py inspect  books/<name>.pdf
src/.venv/bin/python src/pdf_to_text.py convert  books/<name>.pdf books/<name>.txt --first-page N --last-page M --strip-author-year
src/.venv/bin/python src/text_check.py books/<name>.txt --sample 20
```

- Both converters write `books/<name>.report.txt`: removal counts, leftovers (URLs, DOIs, citations, note markers, notes headings, invisible characters) with examples, and 20 random paragraphs. Read it after each run. `text_check.py` gives the same check for any text file.
- `text_check.py` holds the shared code: the leftover patterns, the report writer and `drop_citation_groups()` (used by `--strip-author-year` in both converters). A change there changes both converters.
- `epub_to_text.py` uses the EPUB semantic markup (`epub:type` / `role`: noteref, endnotes, bibliography, pagebreak, epigraph). Spine files are skipped by file name (`SKIP_FILES`). Author-year citations that link to the bibliography are always removed (unless `--keep-citations`).
- `pdf_to_text.py` uses font size and position. It skips "Notes" / "References" / "Bibliography" sections until the next chapter heading (`--chapter-ratio` x body size, or "Chapter N"), and stops only at "Index". It removes running heads by repeat count and by the printed page number. It joins words split at a line end by a soft hyphen (U+00AD) or a zero-width space (U+200B).
- Command-line commands on the Mac: use `src/.venv/bin/python` (it has pymupdf, beautifulsoup4, lxml).

Exam building (lab side), in this order. One work folder per book:

```
python src/exam_builder.py split    --dir work/book1 --book books/book1.txt
python src/exam_builder.py plan     --dir work/book1 --exams dev acceptance --seed <seed1>
python src/exam_builder.py prompts  --dir work/book1 --title "<title>" --templates prompts
python src/exam_builder.py generate --dir work/book1 --model <api-model>
python src/exam_builder.py validate --dir work/book1
python src/exam_builder.py floor    --dir work/book1 --model Qwen/Qwen2.5-3B-Instruct
python src/exam_builder.py select   --dir work/book1 --everyday data/everyday.jsonl --seed <seed2>
```

- `--templates prompts` is necessary. The default looks for the prompt files in `src/`.
- Instead of `generate`, you can paste each `prompts/<name>.txt` into Claude chat and save the reply as `raw/<name>.json`. For sealed books, use only `generate`, and use `--exams acceptance` only.
- `generate` skips batches that already have a `raw/` file, so you can run it again after a failure.
- `plan` takes more items than the exam needs (120 direct, 80 pairs, 80 trap units). `validate` and `floor` remove items, and `select` takes the final 90/60/60/30/15.

## Exam builder data flow

`units.jsonl` (IDs `U0001`...) → `plan.json` (a disjoint unit pool for each exam) → `prompts/*.txt` + `batches.json` → `raw/*.json` → `candidates_<exam>.jsonl` / `rejects_<exam>.jsonl` → `floor_<exam>.jsonl` (removes questions the base model already answers) → `exam_<exam>.jsonl` + `.sha256` + `cue_check_<exam>.json`.

`validate` applies the automatic checks. These checks are the main protection against two known failure modes (traps that share a cue word, a fixed answer template):

- no reference to "the book" / "the text"; no yes/no questions
- evidence quotes must be in the unit
- no 5-word phrase copied from the unit
- trap and twin differ by at most 4 words, and the trap has no added doubt words
- duplicate removal

`select` runs the cue-word check (bag-of-words logistic regression, 5-fold). A score above 0.60 is a warning: rewrite the questions.

## Current state (see `agent.md` sections 5 and 6)

The three open books are converted (`work/<name>/` from `prepare_books.py`). Tables in PDFs are an open problem: rotated pages are skipped for now, and the user wants correct parsing of all table structures in the next iteration (`agent.md` section 5c). The user checks the reports. The next step is the familiarity check (`agent.md` section 6). The old sealed books were removed; new sealed books must be selected by a script before the acceptance run (design section 2.4).

## Fixed settings

Do not change these. They are part of the lab contract:

- Exam answers: default Qwen chat template, single turn, greedy decoding, 256 new tokens maximum.
- The baseline recipe in `book_knowledge_project.md` section 4.1.
- The pass lines and the paired bootstrap in `book_knowledge_project.md` section 5.
- Seeds go in `exam_config.json` before the first run. `src/make_exam_config.py --book NAME:ROLE:SELECT_SEED` writes the first version from the existing files and marks missing values `FILL_IN`.

## Writing style for this user

Write replies and documents in ASD-STE100 Simplified Technical English. Use plain engineering words. Ask a question when a request is ambiguous.
