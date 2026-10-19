#!/usr/bin/env python3
"""Write a first exam_config.json from the files that already exist.

  python src/make_exam_config.py \\
      --book critical-software-studies:book1:21211 \\
      --book organized-crime-and-migration:practice_a:<select seed> \\
      --writer openai:gpt-6.1-sol

--book NAME:ROLE:SELECT_SEED, once for each book (roles: book1, practice_a,
practice_b). The select seed is not saved by exam_builder.py, so you give it here.
--writer PROVIDER:MODEL is used only for a book without generate_log.jsonl
(questions written in a chat).

The script reads, for each book:
  work/<name>/units.jsonl                 hash
  work/familiarity/<name>/plan.json       familiarity seed
  work/familiarity/<name>/floor_*.jsonl   number of questions, text-match count
  work/exams/<name>/plan.json             plan seed
  work/exams/<name>/generate_log.jsonl    question-writer provider and model
  work/exams/<name>/exam_<exam>.jsonl     hash and question counts
Values that are in no file are written as "FILL_IN". The script lists them at
the end. Edit the file and replace each one.

The script does not replace an existing exam_config.json (use --out for a new
name). After the first scored run, do not change the fixed settings.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

FILL = "FILL_IN"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_seed(folder):
    p = folder / "plan.json"
    return json.loads(p.read_text(encoding="utf-8"))["seed"] if p.exists() else FILL


def writer_from_log(folder):
    log = folder / "generate_log.jsonl"
    if not log.exists():
        return None
    used = sorted({(r["provider"], r["model"]) for r in read_jsonl(log) if r.get("complete")})
    if len(used) > 1:
        print(f"WARNING {folder}: more than one question writer in generate_log.jsonl: {used}")
    return [{"provider": p, "model": m} for p, m in used]


def familiarity(work, name):
    folder = work / "familiarity" / name
    floor = sorted(folder.glob("floor_*.jsonl"))
    if not folder.exists():
        return {"seed": FILL, "questions": FILL, "text_match_known": FILL,
                "judged_correct": FILL, "accepted": FILL}
    rows = read_jsonl(floor[0]) if floor else []
    return {
        "seed": plan_seed(folder),
        "questions": len(rows) if rows else FILL,
        "text_match_known": sum(bool(r.get("known")) for r in rows) if rows else FILL,
        "judged_correct": FILL,
        "accepted": FILL,
    }


def book_entry(work, name, role, select_seed, writer, floor_model):
    folder = work / "exams" / name
    units = work / name / "units.jsonl"
    exams = {}
    for f in sorted(folder.glob("exam_*.jsonl")):
        exam = f.stem.replace("exam_", "")
        digest = sha256(f)
        recorded = folder / f"exam_{exam}.sha256"
        if recorded.exists() and recorded.read_text().split()[0] != digest:
            print(f"WARNING {f}: the hash differs from {recorded.name}. The exam file changed after select.")
        exams[exam] = {"file": str(f), "sha256": digest,
                       "counts": dict(Counter(r["set"] for r in read_jsonl(f)))}
    if not exams:
        print(f"WARNING {name}: no exam_*.jsonl in {folder}")
    return {
        "name": name,
        "role": role,
        "units_sha256": sha256(units) if units.exists() else FILL,
        "familiarity": familiarity(work, name),
        "question_writer": writer_from_log(folder) or [writer],
        "plan_seed": plan_seed(folder),
        "select_seed": select_seed,
        "floor_filter_model": floor_model,
        "exams": exams,
    }


def config(a, books):
    return {
        "config_version": 1,
        "created": time.strftime("%Y-%m-%d"),
        "answer_settings": {
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "chat_template": "default",
            "system_prompt": "default",
            "turns": 1,
            "book_text_in_prompt": False,
            "temperature": 0,
            "max_new_tokens": 256,
        },
        "judge": {
            "provider": FILL,
            "model": FILL,
            "temperature": 0,
            "blind": True,
            "hand_check": {"sample_size": 20, "max_disagreements": 2},
        },
        "labels": {
            "knowledge": ["correct", "incorrect"],
            "trap": ["rejected", "accepted", "dodged"],
            "twin": ["answered_correct", "answered_wrong", "refused"],
            "everyday": ["answered", "refused"],
        },
        "exam_counts": {"knowledge_direct": 90, "knowledge_indirect": 60,
                        "trap": 60, "twin": 30, "everyday": 15},
        "everyday_file": {"path": a.everyday,
                          "sha256": sha256(a.everyday) if Path(a.everyday).exists() else FILL},
        "pass_lines": {
            "knowledge_min_gain_points": 8,
            "knowledge_ci_low_must_exceed": 0,
            "trap_accepted_max": 2,
            "trap_dodged_max": 6,
            "twin_refused_max": 2,
            "everyday_refused_max": 2,
            "retention_min_share": 0.95,
        },
        "bootstrap": {"seed": FILL, "resamples": 2000, "interval": 0.95, "rounding": "whole points"},
        "baseline_recipe": {
            "teacher": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "teacher_temperature": 0.7,
            "pairs_per_unit": 5,
            "teacher_prompt": "Write 5 question-answer pairs about this passage.",
            "method": "QLoRA",
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "learning_rate": 0.0002,
            "epochs": 3,
            "effective_batch_size": 16,
            "precision": "fp16",
            "train_from": "base model, one book only",
        },
        "compute_cap_per_book": {"gpu": "T4", "gpu_hours": 5, "wall_clock_hours": 10},
        "books": books,
    }


def find_fill(obj, path=""):
    if obj == FILL:
        yield path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from find_fill(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_fill(v, f"{path}[{i}]")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--book", action="append", required=True, help="NAME:ROLE:SELECT_SEED")
    p.add_argument("--writer", default=f"{FILL}:{FILL}",
                   help="PROVIDER:MODEL, for books without generate_log.jsonl")
    p.add_argument("--work", default="work")
    p.add_argument("--everyday", default="data/everyday.jsonl")
    p.add_argument("--floor-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--out", default="exam_config.json")
    a = p.parse_args()

    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists. It is not replaced. Use --out <other name>.")
    provider, _, model = a.writer.partition(":")
    writer = {"provider": provider, "model": model or FILL}

    books = []
    for spec in a.book:
        parts = spec.split(":")
        if len(parts) != 3 or not parts[2].isdigit():
            sys.exit(f"--book {spec}: use NAME:ROLE:SELECT_SEED, for example "
                     f"critical-software-studies:book1:21211")
        books.append(book_entry(Path(a.work), parts[0], parts[1], int(parts[2]), writer, a.floor_model))

    cfg = config(a, books)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    missing = list(find_fill(cfg))
    if missing:
        print(f"\nReplace these {len(missing)} FILL_IN values:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    main()
