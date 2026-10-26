"""Stage 2: the teacher lists the atomic facts of each unit.

Pass 1 lists the facts of each unit. Pass 2 (optional, config "more") gives the
teacher the unit and the pass-1 facts and asks for the details that are not in
the list yet (names, examples, numbers, quoted terms). The terms pass (optional,
config "terms") is pass 3: the code finds the quoted terms and the names of the
unit that are rare in the book and are not a key yet, and the teacher writes one
fact for each, with the term as the key.
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
# the joined and traps replies, "term" in the terms replies.
MISSING_SPLIT = re.compile(r'(")\s*,(\s*)("(?:fact|n|term)"\s*:)')


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


QUOTED = re.compile(r'"([^"\n]{2,80})"|\u201c([^\u201d\n]{2,80})\u201d|\u2018([^\u2019\n]{2,80})\u2019(?!\w)')
# A run of capitalised words (and numbers) inside a sentence: "Puzzle Pirates", "Lotus 1-2-3",
# "Pieces of Eight". Small linking words are permitted between two capitalised words.
NAME_RUN = re.compile(r"[A-Z][\w\-/.]*(?:\s+(?:(?:of|the|de|la|le|von|van|for|and)\s+)?[A-Z0-9][\w\-/.]*)*")


def words_of(s):
    return " " + " ".join(re.findall(r"\w+", norm(s))) + " "


def term_candidates(text):
    """Quoted terms first, then names of 2 or more words, then names of one word (from the
    second word of a sentence on, so a word is not a name only because it starts a sentence)."""
    quoted = []
    for m in QUOTED.finditer(text):
        t = next(g for g in m.groups() if g).strip(" ,.;:!?")
        if 1 <= len(t.split()) <= 6 and re.search(r"[A-Za-z]", t):
            quoted.append(t)
    multi, single = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        first = re.match(r"\S+\s*", sent)
        rest = sent[first.end():] if first else ""
        for m in NAME_RUN.finditer(rest):
            t = re.sub(r"['\u2019]s$", "", m.group(0).rstrip(".-/"))
            if len(t.split()) >= 2:
                multi.append(t)
            elif len(t) >= 3:
                single.append(t)
    return quoted + multi + single


def unit_terms(units, kept, c):
    """{unit_id: [terms]}: the candidates of each unit that are in at most max_df of the
    units (a rare detail, not a topic of the whole book) and that no kept fact of the unit
    has as its key (or as a part of its key). At most max_per_unit terms for each unit."""
    texts = {u["id"]: words_of(u["text"]) for u in units}
    limit_df = max(1, int(c["max_df"] * len(units)))
    df_cache, out = {}, {}
    for u in units:
        keys = [words_of(f["key"]) for f in kept.get(u["id"], [])]
        terms, seen = [], set()
        for t in term_candidates(u["text"]):
            w = words_of(t)
            if w.strip() == "" or w in seen:
                continue
            seen.add(w)
            if any(w in k or k in w for k in keys):
                continue                                   # already a key, or a part of a key
            if w not in df_cache:
                df_cache[w] = sum(w in x for x in texts.values())
            if df_cache[w] > limit_df:
                continue
            terms.append(t)
            if len(terms) == c["max_per_unit"]:
                break
        if terms:
            out[u["id"]] = terms
    return out


def collect_terms(units, terms, replies, c, kept):
    """Clean the terms replies and add the facts (key = the term, pass 3) to kept."""
    raw, total = [], {"skip": 0, "term_not_in_fact": 0, "bad_form": 0, "not_sentence": 0,
                      "no_quote": 0, "duplicate": 0, "units_no_json": 0}
    for u in units:
        if u["id"] not in terms:
            continue
        ts = terms[u["id"]]
        text, finish = replies[u["id"]]
        items = parse_json_list(text)
        total["units_no_json"] += items is None
        cand = []
        for it in items or []:
            if not (isinstance(it, dict) and isinstance(it.get("term"), int) and 1 <= it["term"] <= len(ts)):
                total["bad_form"] += 1
                continue
            if it.get("skip"):
                total["skip"] += 1
                continue
            term = ts[it["term"] - 1]
            if not isinstance(it.get("fact"), str) or words_of(term) not in words_of(it["fact"]):
                total["term_not_in_fact"] += 1
                continue
            cand.append({"fact": it["fact"], "key": term, "quote": it.get("quote")})
        old = kept.setdefault(u["id"], [])
        seen = {norm(f[k]) for f in old for k in ("key", "fact")}
        facts, dropped = clean_facts(cand, u["text"], c, len(ts), seen)
        for k, v in dropped.items():
            total[k] += v
        raw.append({"unit_id": u["id"], "terms": ts, "reply": text, "finish": finish, "kept": len(facts)})
        old += [{**f, "pass": 3} for f in facts]
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

    tc = c.get("terms")
    if tc:
        terms = unit_terms(units, kept, tc)
        todo = [u for u in units if u["id"] in terms]
        tpl3 = (PROMPTS / tc["prompt"]).read_text(encoding="utf-8").rstrip("\n")
        listed = lambda u: "\n".join(f"{i}. {t}" for i, t in enumerate(terms[u["id"]], 1))
        replies = ask(teacher, todo, lambda u: tpl3.replace("{PASSAGE}", u["text"])
                      .replace("{TERMS}", listed(u)), c, ctx.seed + 200) if todo else {}
        raw3, total3 = collect_terms(units, terms, replies, c, kept)
        write_jsonl(ctx.book_dir / "raw" / "facts_terms.jsonl", raw3)
        n_terms = sum(len(v) for v in terms.values())
        meta["terms"] = {"units": len(terms), "terms": n_terms, "facts": sum(x["kept"] for x in raw3)}
        meta["dropped_terms"] = total3
        print(f"facts terms: {meta['terms']['facts']} facts for {n_terms} terms in {len(terms)} units; removed: "
              + ", ".join(f"{k} {v}" for k, v in total3.items()))

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
