"""Stage 3: training data from the teacher.

Each data part writes data/<part>.jsonl with rows
{"type", "unit_id", "question", "answer"}. The full teacher replies go to raw/.

Parts ("data.parts" in the config, settings in "data.<part>"):
  baseline_qa  E0: the baseline recipe prompt, N question-answer pairs per unit.
  fact_qa      3a: 3 differently worded questions for each fact of stage 2, each
               with a one-sentence answer that contains the key.
  rewrites     3c: each unit rewritten in several ways (one prompt for each way),
               as the answer to a user request.
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
    c = ctx.cfg["data"]["baseline_qa"]
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
    write_jsonl(ctx.book_dir / "data" / "baseline_qa.jsonl", rows)
    write_jsonl(ctx.book_dir / "raw" / "baseline_qa.jsonl", raw)
    return {"baseline_qa": {"rows": len(rows), "units_with_fewer_pairs": short, "replies_cut": cut}}


# A reference to the source inside a sentence ("In the passage, ...", "..., according to the
# text."). It is removed, and the rest of the sentence is kept.
SOURCE_PHRASE = re.compile(r"\s*,?\s*\b(?:according to|as (?:described|stated|mentioned|discussed|explained) in"
                           r"|in|from|within|based on)\s+(?:the|this)\s+(?:passage|text|excerpt|chapter|book)\b\s*,?",
                           re.I)
# A reference to the source that is left after the removal: the pair is dropped. A plain word
# such as "text" (plain text) or "book" (book trade) is allowed.
FORBIDDEN_WORDS = re.compile(r"\bpassage\b|\b(the|this|that)\s+(text|book|chapter|excerpt|author|authors)\b"
                             r"|\baccording to the (passage|text|book|chapter|excerpt|author)", re.I)
YES_NO = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should|may|might)\b", re.I)
# A person named only by a pronoun ("What was he afraid of?"): the question is not clear alone.
PRONOUN = re.compile(r"\b(he|she|him|his|her|hers)\b", re.I)


def drop_source_phrase(s):
    s = SOURCE_PHRASE.sub(" ", s)
    s = re.sub(r"\s+([,.?!;:])", r"\1", re.sub(r"\s+", " ", s)).strip(" ,")
    return s[:1].upper() + s[1:]


def key_in(key, text):
    """True if the key, or all its words of 3 or more letters, are in the text."""
    k, t = norm(key), norm(text)
    if k in t:
        return True
    words = [w for w in re.findall(r"\w+", k) if len(w) >= 3]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}", t) for w in words)


def check_pair(q, a, fact):
    """The reason to drop a pair, or None if it is good."""
    if not (isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()):
        return "bad_form"
    if FORBIDDEN_WORDS.search(q) or FORBIDDEN_WORDS.search(a):
        return "forbidden_word"
    if YES_NO.match(q.strip()):
        return "yes_no"
    if PRONOUN.search(q):
        return "pronoun"
    if norm(a).split()[:6] == norm(q).rstrip("?").split()[:6]:
        return "repeats_question"
    if not key_in(fact["key"], a):
        return "key_not_in_answer"
    return None


def run_json_jobs(ctx, teacher, c, prompts):
    """One teacher reply for each prompt, with retries (new seed) for replies with no valid JSON."""
    replies, todo = {}, list(range(len(prompts)))
    for attempt in range(1 + c["retries"]):
        out = teacher.chat([prompts[i] for i in todo], c["temperature"], c["max_tokens"],
                           seed=ctx.seed + attempt)
        for i, r in zip(todo, out):
            replies[i] = r
        todo = [i for i in todo if parse_json_list(replies[i][0]) is None]
        if not todo:
            break
    return [replies[i] for i in range(len(prompts))]


def fact_qa(ctx, teacher, units):
    c = ctx.cfg["data"]["fact_qa"]
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

    replies = run_json_jobs(ctx, teacher, c, [prompt(j) for j in jobs])
    rows, raw = [], []
    dropped = {"no_json": 0, "fact_missing": 0, "bad_form": 0, "forbidden_word": 0, "yes_no": 0, "pronoun": 0,
               "repeats_question": 0, "key_not_in_answer": 0, "duplicate": 0}
    fixed, seen, per = 0, set(), c["pairs_per_fact"]
    for (uid, fs), (text, finish) in zip(jobs, replies):
        items = parse_json_list(text)
        raw.append({"unit_id": uid, "facts": [f["fact_id"] for f in fs], "reply": text, "finish": finish})
        if items is None:
            dropped["no_json"] += len(fs) * per
            continue
        got = {}
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("fact"), int) and 1 <= it["fact"] <= len(fs):
                got[it["fact"]] = it
        for j, f in enumerate(fs, 1):
            pairs = got.get(j, {}).get("pairs")
            if not isinstance(pairs, list):
                dropped["fact_missing"] += per
                continue
            for k, p in enumerate(pairs[:per], 1):
                q, a = (p.get("q"), p.get("a")) if isinstance(p, dict) else (None, None)
                if isinstance(q, str) and isinstance(a, str):
                    fixed += bool(SOURCE_PHRASE.search(q) or SOURCE_PHRASE.search(a))
                    q, a = drop_source_phrase(q), drop_source_phrase(a)
                why = check_pair(q, a, f)
                if why is None and norm(q) in seen:
                    why = "duplicate"
                if why:
                    dropped[why] += 1
                    continue
                seen.add(norm(q))
                rows.append({"type": f"fact_q{k}", "unit_id": uid, "fact_id": f["fact_id"],
                             "question": q, "answer": a})
    write_jsonl(ctx.book_dir / "data" / "fact_qa.jsonl", rows)
    write_jsonl(ctx.book_dir / "raw" / "fact_qa.jsonl", raw)
    forms = {f"q{k}": sum(r["type"] == f"fact_q{k}" for r in rows) for k in range(1, per + 1)}
    return {"fact_qa": {"rows": len(rows), **forms, "facts_covered": len({r["fact_id"] for r in rows}),
                        "source_phrases_removed": fixed, "teacher_calls": len(jobs),
                        "replies_cut": sum(r["finish"] == "length" for r in raw)},
            "fact_qa_dropped": dropped}


REWRITE = re.compile(r"REQUEST:\s*(.+?)\s*\n\s*TEXT:\s*(.+)", re.S | re.I)


def rewrites(ctx, teacher, units):
    c = ctx.cfg["data"]["rewrites"]
    rows, raw = [], []
    dropped = {"bad_form": 0, "forbidden_word": 0, "too_short": 0}
    fixed = 0
    for name in c["prompts"]:
        tpl = prompt_template(name)
        replies = teacher.chat([tpl.replace("{PASSAGE}", u["text"]) for u in units],
                               c["temperature"], c["max_tokens"])
        style = name.removesuffix(".txt")
        for u, (text, finish) in zip(units, replies):
            raw.append({"unit_id": u["id"], "prompt": name, "reply": text, "finish": finish})
            m = REWRITE.search(text)
            if not m or finish == "length":
                dropped["bad_form"] += 1
                continue
            q = drop_source_phrase(clean(m.group(1)))
            a = "\n".join(drop_source_phrase(line.strip()) if line.strip() else ""
                          for line in m.group(2).strip().splitlines())
            fixed += bool(SOURCE_PHRASE.search(m.group(1)) or SOURCE_PHRASE.search(m.group(2)))
            if FORBIDDEN_WORDS.search(q) or FORBIDDEN_WORDS.search(a):
                dropped["forbidden_word"] += 1
                continue
            if len(a.split()) < c["min_words"]:
                dropped["too_short"] += 1
                continue
            rows.append({"type": style, "unit_id": u["id"], "question": q, "answer": a})
    write_jsonl(ctx.book_dir / "data" / "rewrites.jsonl", rows)
    write_jsonl(ctx.book_dir / "raw" / "rewrites.jsonl", raw)
    words = sorted(len(r["answer"].split()) for r in rows) or [0]
    return {"rewrites": {"rows": len(rows), "teacher_calls": len(raw), "source_phrases_removed": fixed,
                         "words_median": words[len(words) // 2], "words_max": words[-1],
                         "replies_cut": sum(r["finish"] == "length" for r in raw)},
            "rewrites_dropped": dropped}


PARTS = {"baseline_qa": baseline_qa, "fact_qa": fact_qa, "rewrites": rewrites}


def run(ctx):
    parts = ctx.cfg["data"]["parts"]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise SystemExit(f"unknown data.parts {unknown}. Known: {', '.join(PARTS)}")
    units = read_jsonl(ctx.book_dir / "units.jsonl")
    start = time.time()
    teacher = Teacher(ctx.cfg, ctx.seed)
    counts, seconds = {}, {}
    for p in parts:
        t = time.time()
        counts.update(PARTS[p](ctx, teacher, units))
        seconds[p] = round(time.time() - t, 1)
    write_json(ctx.book_dir / "data.meta.json",
               {"parts": parts, "units": len(units), "counts": counts, "seconds_by_part": seconds,
                "seconds": round(time.time() - start, 1)})
    for name, cnt in counts.items():
        print(f"data {name}: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
