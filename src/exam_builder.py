#!/usr/bin/env python3
"""Exam builder for the dummy book-knowledge project (lab side).

Commands, in this order:
  split     Split a clean book text file into units.
  plan      Select units for each exam, with a fixed seed.
  prompts   Write one prompt file for each batch.
  generate  Send the prompt files to an API (optional): Anthropic (default),
            OpenAI (--provider openai) or Gemini (--provider gemini).
            Manual alternative: paste each file from prompts/ into Claude chat
            and save the reply as raw/<same name>.json.
  validate  Parse the replies and run the automatic checks.
  floor     Ask the base model the questions with no book (GPU + vLLM).
            Removes questions that the base model already answers.
  select    Build the final exam files, run the cue-word check, write hashes.

Example (book 1, dev exam + acceptance exam):
  python exam_builder.py split    --dir work/book1 --book books/book1.txt
  python exam_builder.py plan     --dir work/book1 --exams dev acceptance --seed 1101
  python exam_builder.py prompts  --dir work/book1 --title "Book title"
  python exam_builder.py generate --dir work/book1 --model <api-model-name>
  python exam_builder.py generate --dir work/book1 --provider openai --model gpt-6.1-sol
  python exam_builder.py generate --dir work/book1 --provider gemini --model gemini-3.8-flash
  python exam_builder.py validate --dir work/book1
  python exam_builder.py floor    --dir work/book1 --model Qwen/Qwen2.5-3B-Instruct
  python exam_builder.py select   --dir work/book1 --everyday everyday.jsonl --seed 2202

For a sealed book use only one exam (--exams acceptance), use "generate"
(not chat), and do not open the files in its folder.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent

BANNED_PHRASES = [
    "the author", "the text", "this chapter", "this unit", "this book",
    "the book", "according to", "the passage", "this section", "the excerpt",
]
YESNO_START = re.compile(
    r"^(is|are|was|were|did|does|do|has|have|had|can|could|will|would|"
    r"should|shall|may|might|must)\b", re.I)
DOUBT_WORDS = {
    "not", "never", "no", "nor", "allegedly", "supposedly", "reportedly",
    "claimed", "claim", "claims", "rumored", "rumoured", "really", "actually",
    "falsely", "mistakenly", "wrongly", "false", "untrue", "myth",
}
DOUBT_PHRASES = ["is it true", "true that"]


# ---------------------------------------------------------------- helpers

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    for x, y in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-")):
        s = s.replace(x, y)
    return re.sub(r"\s+", " ", s).strip()


def words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", norm(s))


def wc(s):
    return len(s.split())


def chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def load_units(d):
    return {u["id"]: u["text"] for u in read_jsonl(Path(d) / "units.jsonl")}


def unit_index(d):
    return {u["id"]: i for i, u in enumerate(read_jsonl(Path(d) / "units.jsonl"))}


def quote_found(quote, text):
    """True if the quote is in the text (small differences are accepted)."""
    q = norm(quote).strip(" .\"'")
    t = norm(text)
    if len(q) < 8:
        return False
    if q in t:
        return True
    parts = [p.strip(" .\"'") for p in q.split("...") if len(p.strip()) > 10]
    if len(parts) > 1 and all(p in t for p in parts):
        return True
    m = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    return m.size >= 0.85 * len(q)


def shares_ngram(question, text, n):
    qw, tw = words(question), words(text)
    grams = {tuple(tw[i:i + n]) for i in range(len(tw) - n + 1)}
    return any(tuple(qw[i:i + n]) in grams for i in range(len(qw) - n + 1))


def changed_words(a, b):
    wa, wb = words(a), words(b)
    ops = SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def doubt_cue(twin, trap):
    added = Counter(words(trap)) - Counter(words(twin))
    if any(w in DOUBT_WORDS or w.endswith("n't") for w in added):
        return True
    nt, np_ = norm(twin), norm(trap)
    return any(p in np_ and p not in nt for p in DOUBT_PHRASES)


def style_problems(q):
    out = []
    nq = norm(q)
    if any(p in nq for p in BANNED_PHRASES):
        out.append("refers_to_book_or_text")
    if YESNO_START.match(q.strip()):
        out.append("yes_no_question")
    return out


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("not a list")
    return data


def contains_answer(output, answers):
    o = norm(output)
    for v in answers:
        nv = norm(v).strip(" .")
        if len(nv) < 3 and not nv.isdigit():
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(nv) + r"(?![a-z0-9])", o):
            return True
    return False


# ---------------------------------------------------------------- split

def cmd_split(a):
    text = Path(a.book).read_text(encoding="utf-8").replace("\r\n", "\n")
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]

    pieces = []
    for p in paras:
        if wc(p) <= a.max_words:
            pieces.append(p)
            continue
        cur = []
        for s in re.split(r"(?<=[.!?])\s+", p):
            if cur and wc(" ".join(cur + [s])) > a.max_words:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(s)
        if cur:
            pieces.append(" ".join(cur))

    units, cur = [], []
    for p in pieces:
        cur.append(p)
        if wc(" ".join(cur)) >= a.min_words:
            units.append("\n".join(cur))
            cur = []
    if cur:
        if units:
            units[-1] += "\n" + "\n".join(cur)
        else:
            units.append("\n".join(cur))

    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"U{i + 1:04d}", "text": u} for i, u in enumerate(units)]
    write_jsonl(d / "units.jsonl", rows)
    sizes = sorted(wc(u) for u in units)
    print(f"units: {len(rows)}  words: min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
          f"max {sizes[-1]}, total {sum(sizes)}")


# ---------------------------------------------------------------- plan

def make_pairs(pool, n, gap, rng, idx):
    pairs, use, tries = set(), Counter(), 0
    while len(pairs) < n and tries < 200000:
        tries += 1
        x, y = rng.sample(pool, 2)
        if abs(idx[x] - idx[y]) < gap or use[x] >= 2 or use[y] >= 2:
            continue
        p = tuple(sorted((x, y), key=idx.get))
        if p in pairs:
            continue
        pairs.add(p)
        use[x] += 1
        use[y] += 1
    return [list(p) for p in sorted(pairs, key=lambda p: (idx[p[0]], idx[p[1]]))]


def cmd_plan(a):
    d = Path(a.dir)
    idx = unit_index(d)
    ids = list(idx)
    rng = random.Random(a.seed)
    rng.shuffle(ids)
    k = len(a.exams)
    pools = [sorted(ids[i::k], key=idx.get) for i in range(k)]

    plan = {"seed": a.seed, "settings": vars(a).copy(), "exams": {}}
    plan["settings"].pop("func", None)
    for exam, pool in zip(a.exams, pools):
        direct = sorted(rng.sample(pool, min(a.n_direct, len(pool))), key=idx.get)
        traps = sorted(rng.sample(pool, min(a.n_trap, len(pool))), key=idx.get)
        pairs = make_pairs(pool, a.n_pairs, a.min_gap, rng, idx)
        plan["exams"][exam] = {"pool": pool, "direct": direct, "pairs": pairs, "traps": traps}
        print(f"{exam}: pool {len(pool)} units, direct {len(direct)}, "
              f"pairs {len(pairs)}, trap units {len(traps)}")
        if len(direct) < a.n_direct or len(pairs) < a.n_pairs or len(traps) < a.n_trap:
            print("  WARNING: the book is too short for the requested counts.")
    (d / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- prompts

def cmd_prompts(a):
    d = Path(a.dir)
    units, idx = load_units(d), unit_index(d)
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    tdir = Path(a.templates)
    tk = (tdir / "knowledge_prompt.txt").read_text(encoding="utf-8")
    tt = (tdir / "trap_twin_prompt.txt").read_text(encoding="utf-8")
    out = d / "prompts"
    out.mkdir(exist_ok=True)

    def fmt(uids):
        return "\n\n".join(f"[{u}] {units[u]}" for u in uids)

    batches = []
    for exam, p in plan["exams"].items():
        dch = chunks(p["direct"], a.direct_per_batch)
        pch = chunks(p["pairs"], a.pairs_per_batch)
        for i in range(max(len(dch), len(pch))):
            dl = dch[i] if i < len(dch) else []
            pl = pch[i] if i < len(pch) else []
            uids = sorted(set(dl) | {u for pr in pl for u in pr}, key=idx.get)
            name = f"{exam}_knowledge_{i + 1:03d}"
            text = (tk.replace("{BOOK_TITLE}", a.title)
                      .replace("{DIRECT_IDS}", ", ".join(dl) or "(none)")
                      .replace("{PAIRS}", "\n".join(f"{x} + {y}" for x, y in pl) or "(none)")
                      .replace("{N_DIRECT}", str(len(dl)))
                      .replace("{N_INDIRECT}", str(len(pl)))
                      .replace("{UNITS}", fmt(uids)))
            (out / f"{name}.txt").write_text(text, encoding="utf-8")
            batches.append({"name": name, "exam": exam, "kind": "knowledge",
                            "direct_ids": dl, "pairs": pl})
        for i, tl in enumerate(chunks(p["traps"], a.traps_per_batch)):
            name = f"{exam}_traps_{i + 1:03d}"
            text = tt.replace("{BOOK_TITLE}", a.title).replace("{UNITS}", fmt(tl))
            (out / f"{name}.txt").write_text(text, encoding="utf-8")
            batches.append({"name": name, "exam": exam, "kind": "traps", "trap_ids": tl})

    (d / "batches.json").write_text(json.dumps(batches, indent=1), encoding="utf-8")
    (d / "raw").mkdir(exist_ok=True)
    print(f"wrote {len(batches)} prompt files to {out}")


# ---------------------------------------------------------------- generate

# Temperature: --temperature none sends no temperature (the model default).
# Some models reject any value that is not their default, for example
# claude-sonnet-5-5, claude-opus-5-5 and reasoning models of other providers.
DEFAULT_TEMPERATURE = {"anthropic": 0.3, "openai": None, "gemini": None}
DEFAULT_MAX_TOKENS = {"anthropic": 8000, "openai": 16000, "gemini": 16000}


def temperature_arg(v):
    v = v.lower()
    return v if v == "default" else None if v == "none" else float(v)


def call_anthropic(client, a, prompt):
    extra = {} if a.temperature is None else {"temperature": a.temperature}
    msg = client.messages.create(
        model=a.model, max_tokens=a.max_tokens,
        messages=[{"role": "user", "content": prompt}], **extra)
    text = "".join(blk.text for blk in msg.content if blk.type == "text")
    complete = msg.stop_reason == "end_turn"
    return text, msg.usage.input_tokens, msg.usage.output_tokens, complete, msg.stop_reason


def call_openai(client, a, prompt):
    # Reasoning models can reject "temperature": send it only when given.
    # store=False: OpenAI does not keep the prompt and the reply.
    extra = {} if a.temperature is None else {"temperature": a.temperature}
    r = client.responses.create(model=a.model, input=prompt,
                                max_output_tokens=a.max_tokens, store=False, **extra)
    complete = r.status == "completed"
    reason = r.incomplete_details.reason if r.incomplete_details else r.status
    return r.output_text, r.usage.input_tokens, r.usage.output_tokens, complete, reason


def call_gemini(client, a, prompt):
    from google.genai import types
    config = types.GenerateContentConfig(max_output_tokens=a.max_tokens, temperature=a.temperature)
    r = client.models.generate_content(model=a.model, contents=prompt, config=config)
    u = r.usage_metadata
    n_in = (u.prompt_token_count or 0) if u else 0
    n_out = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
    if not r.candidates:                                   # the prompt was blocked
        reason = getattr(r.prompt_feedback, "block_reason", None) or "no reply"
        return "", n_in, n_out, False, str(reason)
    finish = r.candidates[0].finish_reason
    reason = finish.name if finish is not None else "unknown"
    return r.text or "", n_in, n_out, reason == "STOP", reason


def make_api(provider):
    """Return (client, call) for a provider. call(client, a, prompt) needs
    a.model, a.max_tokens and a.temperature, and returns
    (text, input_tokens, output_tokens, complete, reason). Also used by lab_eval.py."""
    if provider == "openai":
        from openai import OpenAI  # pip install openai; needs OPENAI_API_KEY
        return OpenAI(), call_openai
    if provider == "gemini":
        from google import genai  # pip install google-genai; needs GEMINI_API_KEY
        return genai.Client(), call_gemini
    import anthropic  # pip install anthropic; needs ANTHROPIC_API_KEY
    return anthropic.Anthropic(), call_anthropic


def cmd_generate(a):
    if a.max_tokens is None:
        a.max_tokens = DEFAULT_MAX_TOKENS[a.provider]
    if a.temperature == "default":
        a.temperature = DEFAULT_TEMPERATURE[a.provider]
    client, call = make_api(a.provider)
    d = Path(a.dir)
    batches = json.loads((d / "batches.json").read_text(encoding="utf-8"))
    tin = tout = failed = 0
    for b in batches:
        target = d / "raw" / f"{b['name']}.json"
        if target.exists():
            continue
        prompt = (d / "prompts" / f"{b['name']}.txt").read_text(encoding="utf-8")
        text, n_in, n_out, complete, reason = call(client, a, prompt)
        tin += n_in
        tout += n_out
        with open(d / "generate_log.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps({"batch": b["name"], "provider": a.provider, "model": a.model,
                                  "temperature": a.temperature, "max_tokens": a.max_tokens,
                                  "input_tokens": n_in, "output_tokens": n_out,
                                  "complete": complete, "reason": reason,
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        if not complete:
            # a cut-off reply is not saved, so the next run tries this batch again
            failed += 1
            print(f"{b['name']}: NOT SAVED ({reason}). Run again; if the reason is a token "
                  f"limit, use a larger --max-tokens.")
            continue
        target.write_text(text, encoding="utf-8")
        print(f"{b['name']}: done")
    print(f"tokens this run: input {tin}, output {tout}")
    if failed:
        sys.exit(f"{failed} batch(es) not saved")


# ---------------------------------------------------------------- validate

def check_knowledge(it, b, units, a):
    t = it.get("type")
    if t == "skip":
        return "skip", []
    if t not in ("direct", "indirect"):
        return "reject", ["bad_type"]
    q = str(it.get("question", "")).strip()
    ans = str(it.get("answer", "")).strip()
    ids = it.get("unit_ids") or []
    if not q or not ans:
        return "reject", ["missing_field"]
    if t == "direct" and (len(ids) != 1 or ids[0] not in b["direct_ids"]):
        return "reject", ["wrong_unit"]
    if t == "indirect" and (len(ids) != 2 or sorted(ids) not in [sorted(p) for p in b["pairs"]]):
        return "reject", ["wrong_pair"]

    r = style_problems(q)
    if len(words(ans)) > 15:
        r.append("answer_too_long")
    na = norm(ans).strip(" .")
    if len(na) > 3 and na in norm(q):
        r.append("answer_in_question")
    ev = it.get("evidence") or []
    if isinstance(ev, str):
        ev = [ev]
    if not all(any(quote_found(e, units[i]) for e in ev) for i in ids):
        r.append("evidence_not_in_unit")
    if any(shares_ngram(q, units[i], a.ngram) for i in ids):
        r.append("copied_wording")
    return ("reject", r) if r else ("ok", [])


def check_trap(it, b, units, a):
    if it.get("skip"):
        return "skip", []
    uid = it.get("unit_id")
    if uid not in b["trap_ids"]:
        return "reject", ["wrong_unit"]
    need = ["true_fact", "false_fact", "twin_question", "twin_answer",
            "trap_question", "trap_correction", "evidence"]
    if any(not str(it.get(k, "")).strip() for k in need):
        return "reject", ["missing_field"]

    text, tq, pq = units[uid], it["twin_question"], it["trap_question"]
    r = ["twin_" + x for x in style_problems(tq)] + ["trap_" + x for x in style_problems(pq)]
    if norm(tq) == norm(pq):
        r.append("trap_equals_twin")
    elif changed_words(tq, pq) > a.max_diff:
        r.append("trap_twin_diff_too_large")
    if abs(len(words(tq)) - len(words(pq))) > 3:
        r.append("length_mismatch")
    if doubt_cue(tq, pq):
        r.append("doubt_word_in_trap")
    ev = it["evidence"] if isinstance(it["evidence"], list) else [it["evidence"]]
    if not any(quote_found(e, text) for e in ev):
        r.append("evidence_not_in_unit")
    if shares_ngram(tq, text, a.ngram) or shares_ngram(pq, text, a.ngram):
        r.append("copied_wording")
    if len(words(it["twin_answer"])) > 15:
        r.append("answer_too_long")
    return ("reject", r) if r else ("ok", [])


def qa_key(it):
    if it["kind"] == "knowledge":
        return it["question"], it["answer"]
    return it["twin_question"], it["twin_answer"]


def dedupe(items):
    """Drop an item if an earlier item asks the same thing.

    Same thing = same answer and similar wording, or almost the same wording
    about the same unit. Similar wording about different units with different
    answers is not a duplicate.
    """
    keep, dropped = [], 0
    for it in items:
        q, ans = qa_key(it)
        units = set(it.get("unit_ids") or [])
        dup = False
        for k in keep:
            kq, kans = qa_key(k)
            ratio = SequenceMatcher(None, norm(q), norm(kq)).ratio()
            same_unit = bool(units & set(k.get("unit_ids") or []))
            if (norm(ans) == norm(kans) and ratio > 0.6) or (ratio > 0.9 and same_unit):
                dup = True
                break
        if dup:
            dropped += 1
        else:
            keep.append(it)
    return keep, dropped


def cmd_validate(a):
    d = Path(a.dir)
    units = load_units(d)
    batches = json.loads((d / "batches.json").read_text(encoding="utf-8"))
    exams = {}
    for b in batches:
        acc, rej, notes = exams.setdefault(b["exam"], ([], [], Counter()))
        raw = d / "raw" / f"{b['name']}.json"
        if not raw.exists():
            notes["batch_reply_missing"] += 1
            continue
        try:
            items = extract_json(raw.read_text(encoding="utf-8"))
        except Exception:
            notes["batch_reply_unparseable"] += 1
            continue
        for it in items:
            if not isinstance(it, dict):
                notes["item_not_object"] += 1
                continue
            check = check_knowledge if b["kind"] == "knowledge" else check_trap
            status, reasons = check(it, b, units, a)
            if status == "skip":
                notes["skipped_by_writer"] += 1
            elif status == "reject":
                notes.update(reasons)
                rej.append({"batch": b["name"], "reasons": reasons, "item": it})
            else:
                row = dict(it)
                row["batch"] = b["name"]
                if b["kind"] == "knowledge":
                    row["kind"] = "knowledge"
                    row["accepted_variants"] = row.get("accepted_variants") or []
                else:
                    row["kind"] = "trap_pair"
                    row["unit_ids"] = [row["unit_id"]]
                    row["twin_accepted_variants"] = row.get("twin_accepted_variants") or []
                acc.append(row)

    for exam, (acc, rej, notes) in exams.items():
        acc, ndup = dedupe(acc)
        notes["duplicate"] += ndup
        for i, row in enumerate(acc):
            row["cid"] = f"{exam}-C{i + 1:04d}"
        write_jsonl(d / f"candidates_{exam}.jsonl", acc)
        write_jsonl(d / f"rejects_{exam}.jsonl", rej)
        n = Counter((r["kind"], r.get("type", "pair")) for r in acc)
        print(f"{exam}: kept direct {n[('knowledge', 'direct')]}, "
              f"indirect {n[('knowledge', 'indirect')]}, "
              f"trap pairs {n[('trap_pair', 'pair')]}")
        for reason, c in notes.most_common():
            print(f"   {reason}: {c}")


# ---------------------------------------------------------------- floor

def cmd_floor(a):
    from vllm import LLM, SamplingParams  # needs a GPU
    d = Path(a.dir)
    files = sorted(d.glob("candidates_*.jsonl"))
    rows_by_file = {f: read_jsonl(f) for f in files}
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=2048,
              gpu_memory_utilization=a.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=a.max_tokens)

    for f, rows in rows_by_file.items():
        qs = [r["question"] if r["kind"] == "knowledge" else r["twin_question"] for r in rows]
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True)
                   for q in qs]
        outs = llm.generate(prompts, params)
        known = 0
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            if r["kind"] == "knowledge":
                answers = [r["answer"]] + r["accepted_variants"]
            else:
                answers = [r["twin_answer"]] + r["twin_accepted_variants"]
            r["base_answer"] = text
            r["known"] = contains_answer(text, answers)
            known += r["known"]
        exam = f.stem.replace("candidates_", "")
        write_jsonl(d / f"floor_{exam}.jsonl", rows)
        print(f"{exam}: {known} of {len(rows)} answered by the base model (removed in select)")


# ---------------------------------------------------------------- select

def cue_check(exam_rows):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    traps = [r["question"] for r in exam_rows if r["set"] == "trap"]
    twins = [r["question"] for r in exam_rows if r["set"] == "twin"]
    know = [r["question"] for r in exam_rows if r["set"].startswith("knowledge")]
    results = {}
    for name, other in (("traps_vs_twins", twins),
                        ("traps_vs_twins_and_knowledge", twins + know)):
        y = [1] * len(traps) + [0] * len(other)
        model = make_pipeline(CountVectorizer(ngram_range=(1, 2)),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        results[name] = float(cross_val_score(model, traps + other, y, cv=cv,
                                              scoring="balanced_accuracy").mean())
    mean = lambda xs: sum(len(words(x)) for x in xs) / max(len(xs), 1)
    results["mean_words_trap"] = round(mean(traps), 1)
    results["mean_words_twin"] = round(mean(twins), 1)
    results["mean_words_knowledge"] = round(mean(know), 1)
    return results


def cmd_select(a):
    d = Path(a.dir)
    everyday = [r["question"] for r in read_jsonl(a.everyday)][:a.n_everyday]
    if len(everyday) < a.n_everyday:
        sys.exit(f"everyday file has only {len(everyday)} questions")

    for f in sorted(d.glob("candidates_*.jsonl")):
        exam = f.stem.replace("candidates_", "")
        floor = d / f"floor_{exam}.jsonl"
        if floor.exists():
            rows = [r for r in read_jsonl(floor) if not r.get("known")]
        else:
            print(f"{exam}: WARNING no floor file; known questions are not removed")
            rows = read_jsonl(f)

        rng = random.Random(f"{a.seed}-{exam}")
        direct = [r for r in rows if r["kind"] == "knowledge" and r["type"] == "direct"]
        indirect = [r for r in rows if r["kind"] == "knowledge" and r["type"] == "indirect"]
        pairs = [r for r in rows if r["kind"] == "trap_pair"]
        short = [(n, len(x), need) for n, x, need in
                 (("direct", direct, a.n_direct), ("indirect", indirect, a.n_indirect),
                  ("trap pairs", pairs, a.n_trap)) if len(x) < need]
        if short:
            for n, have, need in short:
                print(f"{exam}: not enough {n}: {have} (need {need})")
            print(f"{exam}: generate more batches, then run validate again.")
            continue

        out = []
        for r in rng.sample(direct, a.n_direct):
            out.append({"set": "knowledge_direct", "question": r["question"],
                        "answer": r["answer"], "accepted_variants": r["accepted_variants"],
                        "unit_ids": r["unit_ids"], "cid": r["cid"]})
        for r in rng.sample(indirect, a.n_indirect):
            out.append({"set": "knowledge_indirect", "question": r["question"],
                        "answer": r["answer"], "accepted_variants": r["accepted_variants"],
                        "unit_ids": r["unit_ids"], "cid": r["cid"]})
        for i, r in enumerate(rng.sample(pairs, a.n_trap)):
            pid = f"P{i + 1:03d}"
            out.append({"set": "trap", "question": r["trap_question"],
                        "false_fact": r["false_fact"], "true_fact": r["true_fact"],
                        "correction": r["trap_correction"], "change_type": r.get("change_type"),
                        "unit_ids": r["unit_ids"], "pair_id": pid, "cid": r["cid"]})
            if i < a.n_twin:
                out.append({"set": "twin", "question": r["twin_question"],
                            "answer": r["twin_answer"],
                            "accepted_variants": r["twin_accepted_variants"],
                            "true_fact": r["true_fact"], "unit_ids": r["unit_ids"],
                            "pair_id": pid, "cid": r["cid"]})
        for q in everyday:
            out.append({"set": "everyday", "question": q})

        rng.shuffle(out)
        for i, r in enumerate(out):
            r["qid"] = f"{exam}-{i + 1:03d}"
        path = d / f"exam_{exam}.jsonl"
        write_jsonl(path, out)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (d / f"exam_{exam}.sha256").write_text(f"{digest}  {path.name}\n")

        cues = cue_check(out)
        (d / f"cue_check_{exam}.json").write_text(json.dumps(cues, indent=1))
        counts = Counter(r["set"] for r in out)
        print(f"{exam}: {len(out)} questions {dict(counts)}  sha256 {digest[:12]}")
        for k, v in cues.items():
            flag = "  <-- WARNING (above 0.60)" if k.startswith("traps_vs") and v > 0.60 else ""
            print(f"   {k}: {v:.2f}{flag}" if isinstance(v, float) else f"   {k}: {v}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split")
    s.add_argument("--dir", required=True)
    s.add_argument("--book", required=True)
    s.add_argument("--min-words", type=int, default=150)
    s.add_argument("--max-words", type=int, default=250)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("plan")
    s.add_argument("--dir", required=True)
    s.add_argument("--exams", nargs="+", default=["acceptance"])
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--n-direct", type=int, default=120)
    s.add_argument("--n-pairs", type=int, default=80)
    s.add_argument("--n-trap", type=int, default=80)
    s.add_argument("--min-gap", type=int, default=15, help="minimum distance between units of a pair")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("prompts")
    s.add_argument("--dir", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--templates", default=str(HERE))
    s.add_argument("--direct-per-batch", type=int, default=10)
    s.add_argument("--pairs-per-batch", type=int, default=8)
    s.add_argument("--traps-per-batch", type=int, default=8)
    s.set_defaults(func=cmd_prompts)

    s = sub.add_parser("generate")
    s.add_argument("--dir", required=True)
    s.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default="anthropic")
    s.add_argument("--model", required=True)
    s.add_argument("--max-tokens", type=int,
                   help="output limit, thinking included (default: anthropic 8000, openai and gemini 16000)")
    s.add_argument("--temperature", type=temperature_arg, default="default",
                   help="a number, or 'none' to send no temperature (default: anthropic 0.3, "
                        "openai and gemini none)")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("validate")
    s.add_argument("--dir", required=True)
    s.add_argument("--ngram", type=int, default=5, help="copied-wording limit (words)")
    s.add_argument("--max-diff", type=int, default=4, help="max changed words twin->trap")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("floor")
    s.add_argument("--dir", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--dtype", default="half", help="'half' for T4")
    s.add_argument("--gpu-mem", type=float, default=0.85)
    s.add_argument("--max-tokens", type=int, default=256)
    s.set_defaults(func=cmd_floor)

    s = sub.add_parser("select")
    s.add_argument("--dir", required=True)
    s.add_argument("--everyday", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--n-direct", type=int, default=90)
    s.add_argument("--n-indirect", type=int, default=60)
    s.add_argument("--n-trap", type=int, default=60)
    s.add_argument("--n-twin", type=int, default=30)
    s.add_argument("--n-everyday", type=int, default=15)
    s.set_defaults(func=cmd_select)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
