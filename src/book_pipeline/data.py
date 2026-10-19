"""Stage 3: training data from the teacher.

Each data type writes data/<type>.jsonl with rows
{"type", "unit_id", "question", "answer"}. The full teacher replies go to raw/.

Modes ("data.mode" in the config):
  baseline_qa  E0: the baseline recipe prompt, N question-answer pairs per unit.
"""
import re
import time

from .common import PROMPTS, read_jsonl, write_json, write_jsonl
from .teacher import Teacher

Q_LINE = re.compile(r"^\W*(?:\d+\s*[.)])?\W*(?:Q|Question)\s*\d*\s*[:.)]\s*(.*)$", re.I)
A_LINE = re.compile(r"^\W*(?:A|Answer)\s*\d*\s*[:.)]\s*(.*)$", re.I)


def clean(s):
    return re.sub(r"\s+", " ", s.replace("**", "").replace("__", "")).strip(" *_")


def parse_pairs(text):
    """Question-answer pairs from a reply in the "Q: ... / A: ..." form (and similar forms)."""
    pairs, q, ans, mode = [], None, None, None
    for line in text.splitlines():
        if not line.strip():
            if mode == "a":
                mode = None                  # a blank line ends an answer
            continue
        mq, ma = Q_LINE.match(line), A_LINE.match(line)
        if mq:
            if q and ans:
                pairs.append((clean(q), clean(ans)))
            q, ans, mode = mq.group(1), None, "q"
        elif ma and q is not None:
            ans, mode = ma.group(1), "a"
        elif mode == "q":
            q += " " + line
        elif mode == "a":
            ans += " " + line
    if q and ans:
        pairs.append((clean(q), clean(ans)))
    return [(x, y) for x, y in pairs if x and y]


def prompt_template(name):
    return (PROMPTS / name).read_text(encoding="utf-8").rstrip("\n")


def baseline_qa(ctx, teacher, units):
    c = ctx.cfg["data"]
    tpl = prompt_template(c["prompt"])
    replies = teacher.chat([tpl.replace("{PASSAGE}", u["text"]) for u in units],
                           c["temperature"], c["max_tokens"])
    rows, raw, short, cut = [], [], 0, 0
    n = c["pairs_per_unit"]
    for u, (text, finish) in zip(units, replies):
        got = parse_pairs(text)[:n]
        short += len(got) < n
        cut += finish == "length"
        raw.append({"unit_id": u["id"], "reply": text, "finish": finish, "pairs_found": len(got)})
        rows += [{"type": "qa", "unit_id": u["id"], "question": q, "answer": a} for q, a in got]
    write_jsonl(ctx.book_dir / "data" / "qa.jsonl", rows)
    write_jsonl(ctx.book_dir / "raw" / "qa.jsonl", raw)
    return {"qa": {"rows": len(rows), "units_with_fewer_pairs": short, "replies_cut": cut}}


MODES = {"baseline_qa": baseline_qa}


def run(ctx):
    c = ctx.cfg["data"]
    if c["mode"] not in MODES:
        raise SystemExit(f"unknown data.mode '{c['mode']}'. Known: {', '.join(MODES)}")
    units = read_jsonl(ctx.book_dir / "units.jsonl")
    start = time.time()
    teacher = Teacher(ctx.cfg, ctx.seed)
    counts = MODES[c["mode"]](ctx, teacher, units)
    write_json(ctx.book_dir / "data.meta.json",
               {"mode": c["mode"], "units": len(units), "counts": counts,
                "seconds": round(time.time() - start, 1)})
    for name, cnt in counts.items():
        print(f"data {name}: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
