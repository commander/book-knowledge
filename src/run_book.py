#!/usr/bin/env python3
"""Freelancer-side pipeline: put the knowledge of one book into the model weights.

  python src/run_book.py --book work/<name>/book.txt --state <state dir>

One command per book, no changes between books. state/ carries the merged
model and the replay data from one book to the next. Design and stages:
docs/pipeline_design.md. All settings: pipeline_config.json.

Each stage runs in its own process, so the teacher and the student are never on
the GPU at the same time. A completed stage writes <state>/<book_id>/done/<stage>.
After a disconnect, run the same command again: it skips the completed stages,
and the training continues from its last checkpoint. The time log
(<state>/<book_id>/time_log.json) counts all sessions, also the interrupted ones.

The pipeline reads only the book file, the state folder, its own code, its
prompts (pipeline_prompts/) and its config. It never reads exams or lab output.
"""
import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_pipeline.common import (REPO, Ctx, code_hashes, read_json, sha256_bytes,  # noqa: E402
                                  sha256_file, write_json)
from book_pipeline.timelog import TimeLog  # noqa: E402

STAGES = {"split": "book_pipeline.split", "facts": "book_pipeline.facts", "data": "book_pipeline.data",
          "train": "book_pipeline.train", "save": "book_pipeline.save"}
# Lab-side files. The pipeline must not read them.
FORBIDDEN = re.compile(r"(^|[/_.-])(exam|candidates|rejects|floor|labels|answers|score|hand_check)"
                       r"([/_.-]|$)|(^|/)runs/|exam[-_]config", re.I)


def check_paths(book, state, config):
    for p in (book, state, config):
        if FORBIDDEN.search(str(p.resolve())):
            sys.exit(f"refused: {p} looks like a lab-side file or folder. The pipeline must not read exams.")
    if not book.is_file():
        sys.exit(f"{book} not found")


def make_ctx(a):
    cfg = read_json(a.config)
    book = Path(a.book)
    book_id = sha256_bytes(book.read_bytes())[:12]
    return Ctx(cfg=cfg, book_path=book, state=Path(a.state), book_id=book_id)


def run_stage(a):
    """Inside the child process: run one stage."""
    importlib.import_module(STAGES[a.stage]).run(make_ctx(a))


def check_state(ctx, cfg_hash):
    """One book at a time, and the same config for the whole book."""
    books = read_json(ctx.state / "books.json", [])
    for b in books:
        if b["book_id"] != ctx.book_id and b["status"] != "done":
            sys.exit(f"book {b['book_id']} ({b['file']}) is not done. Finish it first, or use a new state folder.")
    this = next((b for b in books if b["book_id"] == ctx.book_id), None)
    if this is None:
        this = {"book_id": ctx.book_id, "file": ctx.book_path.name, "order": len(books) + 1,
                "config_sha256": cfg_hash, "status": "running"}
        books.append(this)
        write_json(ctx.state / "books.json", books)
    elif this["config_sha256"] != cfg_hash:
        sys.exit("pipeline_config.json changed after this book started. Use the old config, or a new state folder.")
    return books, this


def check_cap(ctx, log):
    cap = ctx.cfg["cap"]
    gpu_limit = cap["gpu_hours"] * 3600 - cap["reserve_exam_minutes"] * 60
    msg = None
    if log.gpu_seconds() > gpu_limit:
        msg = (f"GPU time {log.gpu_seconds() / 3600:.2f} h is above the cap minus the exam reserve "
               f"({gpu_limit / 3600:.2f} h)")
    elif log.wall_seconds() > cap["wall_hours"] * 3600:
        msg = f"wall clock {log.wall_seconds() / 3600:.2f} h is above the cap ({cap['wall_hours']} h)"
    if msg:
        if cap["enforce"]:
            sys.exit(f"CAP EXCEEDED: {msg}. The run fails.")
        print(f"WARNING: {msg} (cap.enforce is false)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--book", required=True, help="the clean book text")
    p.add_argument("--state", required=True, help="state folder, carried from book to book")
    p.add_argument("--config", default=str(REPO / "pipeline_config.json"))
    p.add_argument("--stage", choices=list(STAGES), help=argparse.SUPPRESS)  # used by the child processes
    a = p.parse_args()

    if a.stage:
        return run_stage(a)

    book, state, config = Path(a.book), Path(a.state), Path(a.config)
    check_paths(book, state, config)
    ctx = make_ctx(a)
    unknown = [s for s in ctx.cfg["stages"] if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stages in the config: {unknown}. Known: {list(STAGES)}")
    ctx.book_dir.mkdir(parents=True, exist_ok=True)
    books, this = check_state(ctx, sha256_file(config))
    if this["status"] == "done":
        print(f"book {ctx.book_id} is already done in {state}")
        return
    write_json(ctx.book_dir / "run.json", {"book_file": str(book), "book_sha256": sha256_file(book),
                                           "experiment": ctx.cfg.get("experiment"),
                                           "config_sha256": sha256_file(config), "code": code_hashes()})

    log = TimeLog(ctx.book_dir / "time_log.json")
    print(f"book {ctx.book_id} ({book.name}), order {this['order']}, experiment {ctx.cfg.get('experiment')}, "
          f"start model {ctx.start_model()}")
    for stage in ctx.cfg["stages"]:
        done = ctx.book_dir / "done" / stage
        if done.exists():
            print(f"[{stage}] done before, skipped")
            continue
        check_cap(ctx, log)
        print(f"[{stage}] start")
        log.start(stage)
        rc = subprocess.run([sys.executable, __file__, "--book", str(book), "--state", str(state),
                             "--config", str(config), "--stage", stage]).returncode
        log.end("ok" if rc == 0 else f"failed (exit {rc})")
        if rc != 0:
            sys.exit(f"[{stage}] failed (exit {rc}). Fix the cause, then run the same command again.")
        done.parent.mkdir(exist_ok=True)
        done.touch()

    s = log.summary()
    check_cap(ctx, log)
    this["status"] = "done"
    write_json(state / "books.json", books)
    print("time by stage: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in s["seconds_by_stage"].items()))
    print(f"total GPU time {s['gpu_seconds_total'] / 3600:.2f} h (cap {ctx.cfg['cap']['gpu_hours']} h, "
          f"exam answers not included), wall clock {s['wall_seconds_total'] / 3600:.2f} h")
    print(f"merged model: {ctx.model_dir}")


if __name__ == "__main__":
    main()
