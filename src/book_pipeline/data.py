"""Stage 3: training data from the teacher.

Each data type writes data/<type>.jsonl with rows
{"type", "unit_id", "question", "answer"}. The full teacher replies go to raw/.

Modes ("data.mode" in the config):
  baseline_qa  E0: the baseline recipe prompt, N question-answer pairs per unit.
  fact_qa      E1: 3 question-answer pairs (direct, reworded, reverse) for each
               fact of stage 2, with automatic checks.
"""
import re
import time

from .common import PROMPTS, read_jsonl, write_json, write_jsonl
from .facts import norm, parse_json_list
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


# References to the source ("the text", "this book", "according to the author"). A plain word
# such as "text" (plain text) or "book" (book trade) is allowed.
FORBIDDEN_WORDS = re.compile(r"\bpassage\b|\b(the|this|that)\s+(text|book|chapter|excerpt|author|authors)\b"
                             r"|\baccording to the\b", re.I)
YES_NO = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|may|might)\b", re.I)
KINDS = ("direct", "reworded", "reverse")


def key_in(key, text):
    """True if the key, or all its words of 3 or more letters, are in the text."""
    k, t = norm(key), norm(text)
    if k in t:
        return True
    words = [w for w in re.findall(r"\w+", k) if len(w) >= 3]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}", t) for w in words)


def check_pair(kind, q, a, fact):
    """The reason to drop a pair, or None if it is good."""
    if not (isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()):
        return "bad_form"
    if FORBIDDEN_WORDS.search(q) or FORBIDDEN_WORDS.search(a):
        return "forbidden_word"
    if YES_NO.match(q.strip()):
        return "yes_no"
    if norm(a).split()[:6] == norm(q).rstrip("?").split()[:6]:
        return "repeats_question"
    if kind == "direct" and not key_in(fact["key"], a):
        return "key_not_in_answer"
    if kind == "reverse" and not key_in(fact["key"], q):
        return "key_not_in_question"
    return None


def fact_qa(ctx, teacher, units):
    c = ctx.cfg["data"]
    tpl = prompt_template(c["prompt"])
    text_of = {u["id"]: u["text"] for u in units}
    by_unit = {}
    for f in read_jsonl(ctx.book_dir / "facts.jsonl"):
        by_unit.setdefault(f["unit_id"], []).append(f)
    n = c["facts_per_call"]
    jobs = [(uid, fs[i:i + n]) for uid, fs in by_unit.items() for i in range(0, len(fs), n)]

    def prompt(job):
        uid, fs = job
        listed = "\n".join(f"{i}. {f['fact']} (key: {f['key']})" for i, f in enumerate(fs, 1))
        return tpl.replace("{PASSAGE}", text_of[uid]).replace("{FACTS}", listed)

    replies, todo = {}, list(range(len(jobs)))
    for attempt in range(1 + c["retries"]):           # a retry (new seed) for replies with no valid JSON
        out = teacher.chat([prompt(jobs[i]) for i in todo], c["temperature"], c["max_tokens"],
                           seed=ctx.seed + attempt)
        for i, r in zip(todo, out):
            replies[i] = r
        todo = [i for i in todo if parse_json_list(replies[i][0]) is None]
        if not todo:
            break

    rows, raw = [], []
    dropped = {"no_json": 0, "fact_missing": 0, "bad_form": 0, "forbidden_word": 0, "yes_no": 0,
               "repeats_question": 0, "key_not_in_answer": 0, "key_not_in_question": 0, "duplicate": 0}
    seen = set()
    for i, (uid, fs) in enumerate(jobs):
        text, finish = replies[i]
        items = parse_json_list(text)
        raw.append({"unit_id": uid, "facts": [f["fact_id"] for f in fs], "reply": text, "finish": finish})
        if items is None:
            dropped["no_json"] += len(fs) * len(KINDS)
            continue
        got = {}
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("fact"), int) and 1 <= it["fact"] <= len(fs):
                got[it["fact"]] = it
        for j, f in enumerate(fs, 1):
            it = got.get(j)
            if it is None:
                dropped["fact_missing"] += len(KINDS)
                continue
            for kind in KINDS:
                pair = it.get(kind) if isinstance(it.get(kind), dict) else {}
                q, a = pair.get("q"), pair.get("a")
                why = check_pair(kind, q, a, f)
                if why is None and norm(q) in seen:
                    why = "duplicate"
                if why:
                    dropped[why] += 1
                    continue
                seen.add(norm(q))
                rows.append({"type": f"fact_{kind}", "unit_id": uid, "fact_id": f["fact_id"],
                             "question": q.strip(), "answer": a.strip()})
    write_jsonl(ctx.book_dir / "data" / "fact_qa.jsonl", rows)
    write_jsonl(ctx.book_dir / "raw" / "fact_qa.jsonl", raw)
    kinds = {k: sum(r["type"] == f"fact_{k}" for r in rows) for k in KINDS}
    facts_covered = len({r["fact_id"] for r in rows})
    return {"fact_qa": {"rows": len(rows), **kinds, "facts_covered": facts_covered,
                        "teacher_calls": len(jobs), "replies_cut": sum(r["finish"] == "length" for r in raw)},
            "fact_qa_dropped": dropped}


MODES = {"baseline_qa": baseline_qa, "fact_qa": fact_qa}


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
