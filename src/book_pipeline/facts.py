"""Stage 2: the teacher lists the atomic facts of each unit.

Pass 1 lists the facts of each unit. Pass 2 (optional, config "more") gives the
teacher the unit and the pass-1 facts and asks for the details that are not in
the list yet (names, examples, numbers, quoted terms).
Each fact must have a quote that is in the unit. A fact with no matching quote
is removed: this removes most facts that the teacher invented. Duplicate facts
(the same key in the same unit, or the same fact text) are removed.
Output: facts.jsonl with {"fact_id", "unit_id", "fact", "key", "quote", "pass"}.
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


# A "fact" or "n" key after a comma: the teacher left out the "}, {" between two objects.
# "fact" is always the first key of an object in the facts and fact_qa replies, "n" in
# the joined and traps replies.
MISSING_SPLIT = re.compile(r'(")\s*,(\s*)("(?:fact|n)"\s*:)')


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


def parse_sequence(text):
    """The objects of a reply that has several JSON values one after the other
    ('[{"fact": 1, ...}]\n[{"fact": 2, ...}]' or '...]}{"fact": 2, ...'). Reads a value at
    each [ or { in turn; a value that does not parse is tried again at the next bracket.
    None if fewer than 2 values parse."""
    dec = json.JSONDecoder(object_pairs_hook=no_duplicate_keys)
    t = MISSING_SPLIT.sub(r"\1},\2{\3", escape_inner_quotes(text))
    items, values, i = [], 0, 0
    while True:
        m = re.compile(r"[\[{]").search(t, i)
        if not m:
            break
        try:
            v, i = dec.raw_decode(t, m.start())
        except ValueError:
            i = m.start() + 1
            continue
        values += 1
        items += [x for x in v if isinstance(x, dict)] if isinstance(v, list) else [v]
    return items if values >= 2 and items else None


def parse_json_list(text, return_error=False):
    """The first JSON array in a reply (also inside ```json fences). None if there is no valid array.
    A reply that fails is tried again with the repairs: inner quotes escaped, a missing "}, {"
    added, missing closing brackets added, several values one after the other read in turn.
    With return_error, also the last JSON error."""
    start, end = text.find("["), text.rfind("]")
    error = "no ["
    if start >= 0:
        for c in candidates(text, start, end):
            try:
                data = json.loads(c, object_pairs_hook=no_duplicate_keys)
            except json.JSONDecodeError as e:
                error = f"{e.msg} at {e.pos}: {c[max(0, e.pos - 80):e.pos + 40]!r}"
                if e.msg == "Extra data":              # a full array, then more values
                    data = parse_sequence(text[start:])
                    if data is not None:
                        return (data, None) if return_error else data
                continue
            except ValueError as e:
                error = str(e)
                continue
            data = data if isinstance(data, list) else None
            return (data, None) if return_error else data
        data = parse_sequence(text[start:])
        if data is not None:
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


def clean_facts(items, unit_text, c, limit, seen=()):
    """The valid facts of one reply, at most limit. seen: the keys and fact texts (norm)
    that are already kept for this unit; a fact that repeats one is a duplicate."""
    unit_norm = norm(unit_text)
    out, seen, dropped = [], set(seen), {"bad_form": 0, "not_sentence": 0, "no_quote": 0, "duplicate": 0}
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
    return out[:limit], dropped


def ask(teacher, units, prompt, c, seed):
    """One reply for each unit: {unit_id: (text, finish)}. A retry (new seed) for units with no valid JSON."""
    replies, todo = {}, units
    for attempt in range(1 + c["retries"]):
        out = teacher.chat([prompt(u) for u in todo], c["temperature"], c["max_tokens"], seed=seed + attempt)
        for u, (text, finish) in zip(todo, out):
            replies[u["id"]] = (text, finish)
        todo = [u for u in todo if parse_json_list(replies[u["id"]][0]) is None]
        if not todo:
            break
    return replies


def fact_list(facts):
    return "\n".join(f"- {f['fact']} (key: {f['key']})" for f in facts) or "(none)"


def collect(units, replies, c, limit, kept, n_pass):
    """Clean the replies of one pass and add the facts to kept {unit_id: [facts]}. Returns raw rows and counts."""
    raw, total = [], {"bad_form": 0, "not_sentence": 0, "no_quote": 0, "duplicate": 0, "units_no_json": 0}
    for u in units:
        text, finish = replies[u["id"]]
        items = parse_json_list(text)
        total["units_no_json"] += items is None
        old = kept.setdefault(u["id"], [])
        seen = {norm(f[k]) for f in old for k in ("key", "fact")}
        facts, dropped = clean_facts(items, u["text"], c, limit, seen)
        for k, v in dropped.items():
            total[k] += v
        raw.append({"unit_id": u["id"], "reply": text, "finish": finish, "kept": len(facts)})
        old += [{**f, "pass": n_pass} for f in facts]
    return raw, total


def run(ctx):
    c = ctx.cfg["facts"]
    units = read_jsonl(ctx.book_dir / "units.jsonl")
    tpl = (PROMPTS / c["prompt"]).read_text(encoding="utf-8").rstrip("\n")
    start = time.time()
    teacher = Teacher(ctx.cfg, ctx.seed)

    kept = {}
    replies = ask(teacher, units, lambda u: tpl.replace("{PASSAGE}", u["text"]), c, ctx.seed)
    raw, total = collect(units, replies, c, c["max_per_unit"], kept, 1)
    write_jsonl(ctx.book_dir / "raw" / "facts.jsonl", raw)
    meta = {"units": len(units), "dropped": total}
    print(f"facts pass 1: {sum(x['kept'] for x in raw)} facts; removed: "
          + ", ".join(f"{k} {v}" for k, v in total.items()))

    more = c.get("more")
    if more:
        tpl2 = (PROMPTS / more["prompt"]).read_text(encoding="utf-8").rstrip("\n")
        replies = ask(teacher, units, lambda u: tpl2.replace("{PASSAGE}", u["text"])
                      .replace("{FACTS}", fact_list(kept[u["id"]])), c, ctx.seed + 100)
        raw2, total2 = collect(units, replies, c, more["max_new"], kept, 2)
        write_jsonl(ctx.book_dir / "raw" / "facts_more.jsonl", raw2)
        meta["dropped_more"] = total2
        print(f"facts pass 2: {sum(x['kept'] for x in raw2)} new facts; removed: "
              + ", ".join(f"{k} {v}" for k, v in total2.items()))

    rows = [{"unit_id": u["id"], **f} for u in units for f in kept[u["id"]]]
    for i, r in enumerate(rows, 1):
        r["fact_id"] = f"F{i:05d}"
    write_jsonl(ctx.book_dir / "facts.jsonl", rows)
    per_unit = [len(kept[u["id"]]) for u in units]
    meta.update({"facts": len(rows),
                 "facts_per_unit": {"min": min(per_unit), "median": sorted(per_unit)[len(per_unit) // 2],
                                    "max": max(per_unit)},
                 "seconds": round(time.time() - start, 1)})
    write_json(ctx.book_dir / "facts.meta.json", meta)
    print(f"facts: {len(rows)} facts from {len(units)} units (per unit min {meta['facts_per_unit']['min']}, "
          f"median {meta['facts_per_unit']['median']}, max {meta['facts_per_unit']['max']})")
