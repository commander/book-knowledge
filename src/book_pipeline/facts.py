"""Stage 2: the teacher lists the atomic facts of each unit.

Each fact must have a quote that is in the unit. A fact with no matching quote
is removed: this removes most facts that the teacher invented. Duplicate facts
(the same key in the same unit, or the same fact text) are removed.
Output: facts.jsonl with {"fact_id", "unit_id", "fact", "key", "quote"}.
"""
import difflib
import json
import re
import time

from .common import PROMPTS, read_jsonl, write_json, write_jsonl
from .teacher import Teacher


def norm(s):
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


KEY_AHEAD = re.compile(r'"\w{1,30}"\s*:')


def escape_inner_quotes(s):
    """Escape the double quotes inside string values. The teacher often copies
    quoted words from the book ("critical mass") into a value without a backslash.
    A quote is part of the JSON form only if it opens a string (after [ { : , and
    after a comma only when a key follows) or closes one (before a comma and the
    next string, or before : } ])."""
    out = []
    for i, ch in enumerate(s):
        if ch == '"' and (i == 0 or s[i - 1] != "\\"):
            before = s[:i].rstrip()[-1:]
            after = s[i + 1:].lstrip()
            opens = before in ("[", "{", ":") or (before == "," and KEY_AHEAD.match(s, i))
            closes = after[:1] in (":", "}", "]", "") or re.match(r',\s*["{\[]', after)
            if not (opens or closes):
                out.append('\\"')
                continue
        out.append(ch)
    return "".join(out)


# A "fact" key after a comma: the teacher left out the "}, {" between two objects.
# "fact" is always the first key of an object in the facts and fact_qa replies.
MISSING_SPLIT = re.compile(r'(")\s*,(\s*)("fact"\s*:)')


def balance_brackets(s):
    """Add the closing brackets that the teacher left out: a } before a ] that
    comes too early ("}]]"), and the brackets still open at the end ("}]}").
    Text after the end of the array, or after the last string or bracket, is removed."""
    out, stack, in_str, esc, last = [], [], False, False, 0
    for ch in s:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                last = len(out)
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            while stack and stack[-1] != ch:
                out.insert(len(out) - 1, stack.pop())
            if stack:
                stack.pop()
            last = len(out)
            if not stack:
                return "".join(out)
    if in_str:
        return "".join(out)                    # cut inside a string: no repair
    return "".join(out[:last]) + "".join(reversed(stack))


def candidates(text, start, end):
    """The reply part from the first [ to the last ], then each repair on top of the one
    before. The bracket repair uses the full rest of the reply, because the last ] can
    be inside the array when the outer ] is missing."""
    part = text[start:end + 1] if end > start else ""
    if part:
        yield part
        yield escape_inner_quotes(part)
    joined = MISSING_SPLIT.sub(r"\1},\2{\3", escape_inner_quotes(text[start:]))
    yield balance_brackets(joined)


def no_duplicate_keys(pairs):
    """A key two times in one object means two objects joined into one: not valid."""
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate key in one object: {keys}")
    return dict(pairs)


def parse_json_list(text, return_error=False):
    """The first JSON array in a reply (also inside ```json fences). None if there is no valid array.
    A reply that fails is tried again with the repairs: inner quotes escaped, a missing "}, {"
    added, missing closing brackets added. With return_error, also the last JSON error."""
    start, end = text.find("["), text.rfind("]")
    error = "no ["
    if start >= 0:
        for c in candidates(text, start, end):
            try:
                data = json.loads(c, object_pairs_hook=no_duplicate_keys)
            except json.JSONDecodeError as e:
                error = f"{e.msg} at {e.pos}: {c[max(0, e.pos - 80):e.pos + 40]!r}"
                continue
            except ValueError as e:
                error = str(e)
                continue
            data = data if isinstance(data, list) else None
            return (data, None) if return_error else data
    return (None, error) if return_error else None


def quote_in_unit(quote, unit_norm, min_share):
    """True if the quote is in the unit, or if one continuous part of the unit
    matches at least min_share of the quote (small copy errors by the teacher)."""
    q = norm(quote)
    if len(q.split()) < 3:
        return False
    if q in unit_norm:
        return True
    m = difflib.SequenceMatcher(None, unit_norm, q, autojunk=False).find_longest_match(0, len(unit_norm), 0, len(q))
    return m.size >= min_share * len(q)


def clean_facts(items, unit_text, c):
    unit_norm = norm(unit_text)
    out, seen, dropped = [], set(), {"bad_form": 0, "not_sentence": 0, "no_quote": 0, "duplicate": 0}
    for it in items or []:
        if not (isinstance(it, dict) and all(isinstance(it.get(k), str) and it[k].strip()
                                             for k in ("fact", "key", "quote"))):
            dropped["bad_form"] += 1
            continue
        if len(it["fact"].split()) < 5 or norm(it["fact"]) == norm(it["key"]):
            dropped["not_sentence"] += 1         # a term only ("critical mass"), not a statement
            continue
        if not quote_in_unit(it["quote"], unit_norm, c["quote_min_share"]):
            dropped["no_quote"] += 1
            continue
        k = norm(it["key"])
        if k in seen or norm(it["fact"]) in seen:
            dropped["duplicate"] += 1
            continue
        seen.update({k, norm(it["fact"])})
        out.append({"fact": it["fact"].strip(), "key": it["key"].strip(), "quote": it["quote"].strip()})
    return out[:c["max_per_unit"]], dropped


def run(ctx):
    c = ctx.cfg["facts"]
    units = read_jsonl(ctx.book_dir / "units.jsonl")
    tpl = (PROMPTS / c["prompt"]).read_text(encoding="utf-8").rstrip("\n")
    start = time.time()
    teacher = Teacher(ctx.cfg, ctx.seed)

    replies = {}
    todo = units
    for attempt in range(1 + c["retries"]):       # a retry (new seed) for units with no valid JSON
        out = teacher.chat([tpl.replace("{PASSAGE}", u["text"]) for u in todo],
                           c["temperature"], c["max_tokens"], seed=ctx.seed + attempt)
        for u, (text, finish) in zip(todo, out):
            replies[u["id"]] = (text, finish)
        todo = [u for u in todo if parse_json_list(replies[u["id"]][0]) is None]
        if not todo:
            break

    rows, raw, total = [], [], {"bad_form": 0, "not_sentence": 0, "no_quote": 0, "duplicate": 0,
                                "units_no_json": 0}
    for u in units:
        text, finish = replies[u["id"]]
        items = parse_json_list(text)
        total["units_no_json"] += items is None
        facts, dropped = clean_facts(items, u["text"], c)
        for k, v in dropped.items():
            total[k] += v
        raw.append({"unit_id": u["id"], "reply": text, "finish": finish, "kept": len(facts)})
        rows += [{"unit_id": u["id"], **f} for f in facts]
    for i, r in enumerate(rows, 1):
        r["fact_id"] = f"F{i:05d}"
    write_jsonl(ctx.book_dir / "facts.jsonl", rows)
    write_jsonl(ctx.book_dir / "raw" / "facts.jsonl", raw)
    per_unit = [x["kept"] for x in raw]
    meta = {"units": len(units), "facts": len(rows), "dropped": total,
            "facts_per_unit": {"min": min(per_unit), "median": sorted(per_unit)[len(per_unit) // 2],
                               "max": max(per_unit)},
            "seconds": round(time.time() - start, 1)}
    write_json(ctx.book_dir / "facts.meta.json", meta)
    print(f"facts: {len(rows)} facts from {len(units)} units (per unit min {meta['facts_per_unit']['min']}, "
          f"median {meta['facts_per_unit']['median']}, max {meta['facts_per_unit']['max']}); removed: "
          + ", ".join(f"{k} {v}" for k, v in total.items()))
