#!/usr/bin/env python3
"""Lab-side diagnosis of one pipeline run. Not part of the pipeline: the pipeline never imports it.

  python src/diagnose_run.py data     --state work/state-e1
  python src/diagnose_run.py coverage --state work/state-e1 --run pipeline-e1 --book critical-software-studies --exam dev
  python src/diagnose_run.py recall   --state work/state-e1 --n 300          (GPU, vLLM)
  python src/diagnose_run.py traps    --state work/state-e5

data      Why the facts and data stages lost units and pairs: the facts (all passes) and fact_qa
          replies with no JSON (before and after the repairs), examples of pass-2 and terms facts, joined and
          trap rows, the words that the forbidden-word check found, examples of reverse
          questions without the key.
coverage  For each knowledge item of the exam: is its answer in the training data of
          its unit? Joined with the judge labels. This separates "the data does not
          have the fact" from "the model did not learn the fact". Refuses sealed books.
recall    Asks the trained model a sample of its own training questions (exam answer
          settings: chat template, greedy, 256 tokens) and counts the answers that
          contain the fact key or start with the trained answer. A low share means the model did not learn its data.
traps     Runs the trap checks again on the saved teacher replies and prints
          examples of each reason for a removed trap. No GPU.

The word match is rough: a gold answer is "in" a text if all its words of 3 or more
letters (stop words not counted) are in the text.
"""
import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lab_eval import find_book, load_config, load_exam, run_dir  # noqa: E402
from exam_builder import read_jsonl  # noqa: E402
from book_pipeline.facts import parse_json_list  # noqa: E402  (the pipeline parse, with the quote repair)

STOP = set("the and for with from that this was were are has have had not but its into about "
           "than then them they their there which who whom what when where why how also".split())
# The same pattern as the pipeline check (src/book_pipeline/data.py).
FORBIDDEN_WORDS = re.compile(r"\bpassage\b|\b(the|this|that)\s+(text|book|chapter|excerpt|author|authors)\b"
                             r"|\baccording to the\b", re.I)


def norm(s):
    s = (s or "").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def words(s):
    return [w for w in re.findall(r"\w+", norm(s)) if len(w) >= 3 and w not in STOP]


def contains(answer, text):
    """True if the answer, or all its content words, are in the text."""
    a, t = norm(answer), norm(text)
    if a and a in t:
        return True
    ws = words(answer)
    return bool(ws) and all(re.search(rf"\b{re.escape(w)}", t) for w in ws)


def book_dir(state):
    books = [d for d in Path(state).iterdir() if (d / "units.jsonl").exists()]
    if len(books) != 1:
        sys.exit(f"{state}: expected one book folder, found {len(books)}. This tool handles one book.")
    return books[0]


def load_data(d):
    rows = []
    for f in sorted((d / "data").glob("*.jsonl")):
        rows += read_jsonl(f)
    return rows


def parse_raw(text):
    """The JSON array of a reply, without the repair of the inner quotes (E1 and E2 parse)."""
    import json
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def show_no_json(name, raw, id_key, rng, show):
    """Count the replies with no JSON before and after the repair, and print examples of the lost ones."""
    before = [r for r in raw if parse_raw(r["reply"]) is None]
    lost = [r for r in raw if parse_json_list(r["reply"]) is None]
    print(f"{name}: {len(raw)} replies, no JSON before repair {len(before)}, after repair {len(lost)}")
    print("  finish reason of the lost replies: " + str(dict(Counter(r["finish"] for r in lost))))
    print("  reply length in characters (lost): " + str(sorted(len(r["reply"]) for r in lost)[::max(1, len(lost) // 10)]))
    for r in rng.sample(lost, min(show, len(lost))):
        print(f"  --- {r[id_key]} finish {r['finish']}, start of reply:\n  {r['reply'][:300]!r}\n"
              f"  end of reply:\n  {r['reply'][-200:]!r}\n"
              f"  error after the repairs: {parse_json_list(r['reply'], return_error=True)[1]}")


# ---------------------------------------------------------------- data

def cmd_data(a):
    d = book_dir(a.state)
    rng = random.Random(0)

    raw = read_jsonl(d / "raw" / "facts.jsonl")
    show_no_json("facts", raw, "unit_id", rng, a.show)
    print(f"  units with 0 facts kept: {sum(r['kept'] == 0 for r in raw)}")
    if (d / "raw" / "facts_more.jsonl").exists():
        raw2 = read_jsonl(d / "raw" / "facts_more.jsonl")
        show_no_json("facts pass 2", raw2, "unit_id", rng, a.show)
        facts = read_jsonl(d / "facts.jsonl")
        new = [f for f in facts if f.get("pass") == 2]
        print(f"  new facts per unit: {dict(sorted(Counter(r['kept'] for r in raw2).items()))}")
        print(f"  units with 0 facts after both passes: "
              f"{len({r['unit_id'] for r in raw2} - {f['unit_id'] for f in facts})}")
        print(f"pass 2 facts: {len(new)}. Examples:")
        for f in rng.sample(new, min(a.show, len(new))):
            print(f"  ({f['unit_id']}) {f['fact']}  (key: {f['key']})")
    if (d / "raw" / "facts_terms.jsonl").exists():
        raw3 = read_jsonl(d / "raw" / "facts_terms.jsonl")
        show_no_json("facts terms", raw3, "unit_id", rng, a.show)
        facts = read_jsonl(d / "facts.jsonl")
        new = [f for f in facts if f.get("pass") == 3]
        print(f"  terms per unit: {dict(sorted(Counter(len(r['terms']) for r in raw3).items()))}, "
              f"facts kept per unit: {dict(sorted(Counter(r['kept'] for r in raw3).items()))}")
        got = {(f["unit_id"], f["key"]) for f in new}
        missed = [t for r in raw3 for t in r["terms"] if (r["unit_id"], t) not in got]
        print(f"  terms with no fact: {len(missed)}. Examples: "
              + "; ".join(rng.sample(missed, min(3 * a.show, len(missed)))))
        print(f"terms facts: {len(new)}. Examples:")
        for f in rng.sample(new, min(2 * a.show, len(new))):
            print(f"  ({f['unit_id']}) {f['fact']}  (key: {f['key']})")
    show_no_json("fact_qa", read_jsonl(d / "raw" / "fact_qa.jsonl"), "unit_id", rng, a.show)
    if (d / "raw" / "joined.jsonl").exists():
        show_no_json("joined", read_jsonl(d / "raw" / "joined.jsonl"), "pairs", rng, a.show)
        rows = read_jsonl(d / "data" / "joined.jsonl")
        facts = {f["fact_id"]: f for f in read_jsonl(d / "facts.jsonl")}
        print(f"joined rows: {len(rows)}. Examples:")
        for r in rng.sample(rows, min(a.show, len(rows))):
            print(f"  FACT A ({r['context_unit_id']}): {facts[r['context_fact_id']]['fact']}\n"
                  f"  FACT B ({r['unit_id']}): {facts[r['fact_id']]['fact']}  (key: {facts[r['fact_id']]['key']})\n"
                  f"  Q: {r['question']}\n  A: {r['answer']}\n")
    if (d / "raw" / "traps.jsonl").exists():
        show_no_json("traps", read_jsonl(d / "raw" / "traps.jsonl"), "unit_id", rng, a.show)
        rows = read_jsonl(d / "data" / "traps.jsonl")
        print(f"trap rows: {len(rows)}. Examples:")
        for r in rng.sample(rows, min(a.show, len(rows))):
            print(f"  TWIN Q: {r.get('twin_question')}\n  TRAP Q: {r['question']}\n  TRAP A: {r['answer']}\n")

    facts = {f["fact_id"]: f for f in read_jsonl(d / "facts.jsonl")}
    hits_q, hits_a, no_key = Counter(), Counter(), []
    for r in read_jsonl(d / "raw" / "fact_qa.jsonl"):
        items = parse_json_list(r["reply"]) or []
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("fact"), int) or not 1 <= it["fact"] <= len(r["facts"]):
                continue
            f = facts[r["facts"][it["fact"] - 1]]
            if isinstance(it.get("pairs"), list):        # E3 form
                pairs = [("q", p) for p in it["pairs"] if isinstance(p, dict)]
            else:                                        # E1 form
                pairs = [(k, it[k]) for k in ("direct", "reworded", "reverse") if isinstance(it.get(k), dict)]
            for kind, p in pairs:
                q, ans = p.get("q") or "", p.get("a") or ""
                if not isinstance(q, str) or not isinstance(ans, str):
                    continue
                hits_q.update(m.group(0).lower() for m in FORBIDDEN_WORDS.finditer(q))
                hits_a.update(m.group(0).lower() for m in FORBIDDEN_WORDS.finditer(ans))
                if kind == "reverse" and q and not contains(f["key"], q):
                    no_key.append((f["key"], q))
    print(f"forbidden words in questions: {dict(hits_q.most_common(10))}")
    print(f"forbidden words in answers:   {dict(hits_a.most_common(10))}")
    fact_hits = Counter(m.group(0).lower() for f in facts.values() for m in FORBIDDEN_WORDS.finditer(f["fact"]))
    print(f"forbidden words in the facts themselves: {dict(fact_hits.most_common(10))}")
    print(f"reverse questions without the key (E1 form only): {len(no_key)}. Examples:")
    for key, q in rng.sample(no_key, min(a.show, len(no_key))):
        print(f"  KEY: {key}\n  Q:   {q}")


# ---------------------------------------------------------------- coverage

def cmd_coverage(a):
    cfg = load_config(a.config)
    book = find_book(cfg, a.book)
    if book.get("sealed"):
        sys.exit("This is a sealed book. Do not diagnose with its exam.")
    d = book_dir(a.state)
    rows = [r for r in load_exam(book, a.exam) if r["set"].startswith("knowledge")]
    labels = {r["qid"]: r["label"] for r in read_jsonl(run_dir(a) / f"labels_{a.exam}.jsonl")}
    answers = {r["qid"]: r["response"] for r in read_jsonl(run_dir(a) / f"answers_{a.exam}.jsonl")}
    train = load_data(d)
    facts = read_jsonl(d / "facts.jsonl") if (d / "facts.jsonl").exists() else []
    by_unit, facts_by_unit = {}, Counter(f["unit_id"] for f in facts)
    for t in train:
        by_unit.setdefault(t["unit_id"], []).append(t)

    table, source, examples = Counter(), Counter(), []
    for r in rows:
        gold = [r["answer"]] + (r.get("accepted_variants") or [])
        units = r.get("unit_ids") or []
        mine = [t for u in units for t in by_unit.get(u, [])]
        hits = [t for t in mine if any(contains(g, t["question"] + " " + t["answer"]) for g in gold)]
        qa_hits = [t for t in hits if not t["type"].startswith("rewrite")]
        hit = (qa_hits or hits or [None])[0]
        if hit:
            where = "in unit data"
            source[(r["set"].replace("knowledge_", ""), "in question rows" if qa_hits else "in rewrites only",
                    labels.get(r["qid"], "missing"))] += 1
        elif any(any(contains(g, t["question"] + " " + t["answer"]) for g in gold) for t in train):
            where = "elsewhere in data"
        elif not mine:
            where = "unit has no data"
        else:
            where = "not in data"
        lab = labels.get(r["qid"], "missing")
        table[(r["set"].replace("knowledge_", ""), where, lab)] += 1
        examples.append((where, lab, r, hit, sum(facts_by_unit[u] for u in units)))

    print(f"{a.run} / {a.book} / {a.exam}: training data {len(train)} rows "
          f"({dict(Counter(t['type'] for t in train))})")
    for s in ("direct", "indirect"):
        print(f"  {s}:")
        for where in ("in unit data", "elsewhere in data", "not in data", "unit has no data"):
            c = {lab: n for (ss, w, lab), n in table.items() if ss == s and w == where}
            print(f"    {where:18} total {sum(c.values()):3}  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
            if where == "in unit data":
                for src in ("in question rows", "in rewrites only"):
                    c = {lab: n for (ss, w, lab), n in source.items() if ss == s and w == src}
                    print(f"      {src:16} total {sum(c.values()):3}  "
                          + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    rng = random.Random(0)
    for want in ("in unit data", "not in data"):
        pick = [e for e in examples if e[0] == want and e[1] != "correct"]
        print(f"\n=== examples: answer {want}, judged not correct ({len(pick)})")
        for where, lab, r, hit, nf in rng.sample(pick, min(a.show, len(pick))):
            print(f"--- {r['qid']} [{r['set']}] units {r.get('unit_ids')} ({nf} facts)\n"
                  f"QUESTION: {r['question']}\nKEY: {r['answer']}\n"
                  f"RESPONSE: {answers.get(r['qid'], '')[:300]}")
            if hit:
                print(f"TRAINING ROW: Q: {hit['question']}\n              A: {hit['answer']}")


# ---------------------------------------------------------------- traps

def cmd_traps(a):
    from book_pipeline.data import make_trap
    d = book_dir(a.state)
    cfg = __import__("json").loads((Path(a.config)).read_text(encoding="utf-8"))["data"]["traps"]
    facts = {f["fact_id"]: f for f in read_jsonl(d / "facts.jsonl")}
    text_of = {u["id"]: u["text"] for u in read_jsonl(d / "units.jsonl")}
    by_reason = {}
    for r in read_jsonl(d / "raw" / "traps.jsonl"):
        got = {it["n"]: it for it in parse_json_list(r["reply"]) or []
               if isinstance(it, dict) and isinstance(it.get("n"), int)}
        for j, t in enumerate(r["twins"], 1):
            it = got.get(j, {})
            why, made = make_trap(it, t, facts[t["fact_id"]], text_of[t["unit_id"]], cfg["max_word_diff"])
            by_reason.setdefault(why or "kept", []).append((t, it, made))
    print("traps by check result: " + str({k: len(v) for k, v in sorted(by_reason.items())}))
    rng = random.Random(0)
    for reason, got in sorted(by_reason.items()):
        print(f"\n=== {reason} ({len(got)})")
        for t, it, made in rng.sample(got, min(a.show, len(got))):
            print(f"--- TWIN Q: {t['question']}\n  TWIN A: {t['answer']}  (key: {facts[t['fact_id']]['key']})\n"
                  f"  detail: {it.get('detail')!r}   replacement: {it.get('replacement')!r}"
                  + (f"\n  TRAP Q: {made[2]}" if made else ""))


# ---------------------------------------------------------------- recall

def cmd_recall(a):
    d = book_dir(a.state)
    model = a.model or str(Path(a.state) / "model")
    rows = load_data(d)
    keys = {f["fact_id"]: f["key"] for f in read_jsonl(d / "facts.jsonl")} if (d / "facts.jsonl").exists() else {}
    sample = random.Random(0).sample(rows, min(a.n, len(rows)))

    from vllm import LLM, SamplingParams  # needs a GPU
    llm = LLM(model=model, dtype="half", max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                       tokenize=False, add_generation_prompt=True) for r in sample]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=256))

    got, examples = Counter(), []
    for r, o in zip(sample, outs):
        resp = o.outputs[0].text
        # The fact key when there is one, otherwise the training answer.
        # A trap answer must give the true detail; other rows the fact key, or the training answer.
        target = r.get("true_detail") if r["type"] == "trap" else keys.get(r.get("fact_id")) or r["answer"]
        if r.get("fact_id"):
            ok = contains(target, resp) or norm(resp).startswith(norm(r["answer"]))
        else:   # a long text: at least half of its content words are in the response
            ws = set(words(r["answer"]))
            ok = bool(ws) and len(ws & set(words(resp))) >= len(ws) / 2
        got[(r["type"], ok)] += 1
        examples.append((ok, r, target, resp))
    print(f"recall of training questions, model {model}, {len(sample)} of {len(rows)} rows:")
    for t in sorted({k[0] for k in got}):
        n_ok, n = got[(t, True)], got[(t, True)] + got[(t, False)]
        print(f"  {t:16} {n_ok:4} of {n:4} = {n_ok / n:.0%}")
    total = sum(v for (t, ok), v in got.items() if ok)
    print(f"  {'all':16} {total:4} of {len(sample):4} = {total / len(sample):.0%}")
    rng = random.Random(1)
    wrong = [e for e in examples if not e[0]]
    print(f"\n=== examples of missed training questions ({len(wrong)})")
    for ok, r, target, resp in rng.sample(wrong, min(a.show, len(wrong))):
        print(f"--- [{r['type']}] Q: {r['question']}\nTRAINED A: {r['answer']}\nKEY: {target}\nRESPONSE: {resp[:300]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data")
    s.add_argument("--state", required=True)
    s.add_argument("--show", type=int, default=5, help="number of examples to print")
    s.set_defaults(func=cmd_data)
    s = sub.add_parser("coverage")
    s.add_argument("--state", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--book", required=True)
    s.add_argument("--exam", default="dev")
    s.add_argument("--config", default="exam_config.json")
    s.add_argument("--work", default="work")
    s.add_argument("--show", type=int, default=8)
    s.set_defaults(func=cmd_coverage)
    s = sub.add_parser("traps")
    s.add_argument("--state", required=True)
    s.add_argument("--config", default="pipeline_config.json")
    s.add_argument("--show", type=int, default=4)
    s.set_defaults(func=cmd_traps)
    s = sub.add_parser("recall")
    s.add_argument("--state", required=True)
    s.add_argument("--model", help="default: <state>/model")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--show", type=int, default=8)
    s.set_defaults(func=cmd_recall)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
