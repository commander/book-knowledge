"""Stage 1: split the book into units of min_words..max_words at paragraph limits.

The same method as the lab splitter, but a separate copy: a change on the lab
side must not change the pipeline.
"""
import re

from .common import write_jsonl


def wc(s):
    return len(s.split())


def split_text(text, min_words, max_words):
    text = text.replace("\r\n", "\n")
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]

    pieces = []
    for p in paras:
        if wc(p) <= max_words:
            pieces.append(p)
            continue
        cur = []
        for s in re.split(r"(?<=[.!?])\s+", p):
            if cur and wc(" ".join(cur + [s])) > max_words:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(s)
        if cur:
            pieces.append(" ".join(cur))

    units, cur = [], []
    for p in pieces:
        cur.append(p)
        if wc(" ".join(cur)) >= min_words:
            units.append("\n".join(cur))
            cur = []
    if cur:
        if units:
            units[-1] += "\n" + "\n".join(cur)
        else:
            units.append("\n".join(cur))
    return units


def run(ctx):
    c = ctx.cfg["split"]
    units = split_text(ctx.book_path.read_text(encoding="utf-8"), c["min_words"], c["max_words"])
    write_jsonl(ctx.book_dir / "units.jsonl", [{"id": f"U{i + 1:04d}", "text": u} for i, u in enumerate(units)])
    sizes = sorted(wc(u) for u in units)
    print(f"split: {len(units)} units, words min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
          f"max {sizes[-1]}, total {sum(sizes)}")
