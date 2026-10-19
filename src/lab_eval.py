#!/usr/bin/env python3
"""Lab-side evaluation: answer an exam, judge the answers, score them.

All settings come from exam_config.json (answer settings, judge, labels, pass
lines, bootstrap seed). Exam files are found through the "books" entries of the
config, and their SHA-256 must match the config.

Output for each run, in work/runs/<run>/<book>/:
  answers_<exam>.jsonl      one response for each qid
  answers_<exam>.meta.json  model, adapter, settings, GPU, time
  labels_<exam>.jsonl       one judge label for each qid
  labels_<exam>.meta.json   judge provider, model, prompt hashes, tokens
  score_<exam>.json         counts, pass lines, bootstrap (score command)

Commands, in order:
  answer      A model answers an exam (GPU, vLLM). The model sees only the question.
  judge       An API model labels each answer, blind (no run or model name).
  score       Counts, knowledge score, pass lines 1-3. With --baseline: the
              paired bootstrap against that run (pass line 1).
  retention   Pass line 4: of the book 1 questions that were correct in one run,
              the share that is still correct in a later run.
  hand-check  Print random judged answers for a check by a person.

Examples:
  python src/lab_eval.py answer   --book critical-software-studies --exam dev --run floor
  python src/lab_eval.py judge    --book critical-software-studies --exam dev --run floor
  python src/lab_eval.py score    --book critical-software-studies --exam dev --run floor
  python src/lab_eval.py answer   --book critical-software-studies --exam dev --run baseline \\
                                  --adapter work/runs/baseline/critical-software-studies/adapter
  python src/lab_eval.py score    --book critical-software-studies --exam dev --run pipeline-v1 --baseline baseline
  python src/lab_eval.py retention --book critical-software-studies --exam dev --before after-book1 --after after-book3
  python src/lab_eval.py hand-check --book critical-software-studies --exam dev --run floor
"""
import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exam_builder import make_api, read_jsonl, temperature_arg, write_jsonl  # noqa: E402

FILL = "FILL_IN"
JUDGE_PROMPTS = {"knowledge": "judge_knowledge.txt", "trap": "judge_trap.txt",
                 "twin": "judge_twin.txt", "everyday": "judge_everyday.txt"}
JUDGE_MAX_TOKENS = {"anthropic": 1000, "openai": 4000, "gemini": 4000}


# ---------------------------------------------------------------- config and files

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"{p} not found. Make it with src/make_exam_config.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def need(value, name):
    if value in (None, FILL) or (isinstance(value, str) and FILL in value):
        sys.exit(f"exam_config.json: {name} is not filled in.")
    return value


def find_book(cfg, name):
    for b in cfg["books"]:
        if b["name"] == name:
            return b
    sys.exit(f"book '{name}' is not in exam_config.json. Books: {[b['name'] for b in cfg['books']]}")


def load_exam(book, exam):
    """Exam rows sorted by qid. Stops if the file differs from the config hash."""
    if exam not in book.get("exams", {}):
        sys.exit(f"{book['name']} has no exam '{exam}' in exam_config.json "
                 f"(exams: {list(book.get('exams', {}))})")
    e = book["exams"][exam]
    path = Path(e["file"])
    if not path.exists():
        sys.exit(f"exam file not found: {path}")
    if sha256(path) != need(e.get("sha256"), f"{book['name']}.exams.{exam}.sha256"):
        sys.exit(f"{path} does not match its SHA-256 in exam_config.json. "
                 "The exam changed. Do not score with it.")
    return sorted(read_jsonl(path), key=lambda r: r["qid"])


def run_dir(a, run=None):
    return Path(a.work) / "runs" / (run or a.run) / a.book


def group(set_name):
    return "knowledge" if set_name.startswith("knowledge") else set_name


def versions(*names):
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- answer

def cmd_answer(a):
    cfg = load_config(a.config)
    book = find_book(cfg, a.book)
    rows = load_exam(book, a.exam)
    s = cfg["answer_settings"]
    model = a.model or s["base_model"]
    out_dir = run_dir(a)
    out = out_dir / f"answers_{a.exam}.jsonl"
    if out.exists() and not a.force:
        sys.exit(f"{out} exists. Use --force to answer again (the old file is kept as a backup).")
    if out.exists():
        out.rename(out.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.bak"))
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    from vllm import LLM, SamplingParams  # needs a GPU
    kw = dict(model=model, dtype=a.dtype, max_model_len=2048, gpu_memory_utilization=a.gpu_mem, seed=0)
    lora = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        rank = json.loads((Path(a.adapter) / "adapter_config.json").read_text())["r"]
        kw.update(enable_lora=True, max_lora_rank=max(16, rank))
        lora = LoRARequest("adapter", 1, str(a.adapter))
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    # Default chat template and default system prompt: only the user turn is given.
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    params = SamplingParams(temperature=s["temperature"], max_tokens=s["max_new_tokens"])
    outs = llm.generate(prompts, params, lora_request=lora)
    write_jsonl(out, [{"qid": r["qid"], "set": r["set"], "response": o.outputs[0].text}
                      for r, o in zip(rows, outs)])

    adapter_file = Path(a.adapter or "") / "adapter_model.safetensors"
    meta = {"run": a.run, "book": a.book, "exam": a.exam,
            "exam_sha256": book["exams"][a.exam]["sha256"],
            "model": model, "adapter": a.adapter,
            "adapter_sha256": sha256(adapter_file) if a.adapter and adapter_file.exists() else None,
            "answer_settings": s, "dtype": a.dtype, "gpu": gpu_name(),
            "versions": versions("vllm", "torch", "transformers"),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - start, 1)}
    (out_dir / f"answers_{a.exam}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{a.run}/{a.book}/{a.exam}: {len(rows)} answers in {meta['seconds']} s on {meta['gpu']}")
    print(f"answers: {out}")


# ---------------------------------------------------------------- judge

def judge_prompt(templates, row, response):
    g = group(row["set"])
    t = templates[g]
    key = row.get("answer", "")
    variants = "; ".join(row.get("accepted_variants") or []) or "(none)"
    false_claim = row.get("false_fact", "")
    if row.get("correction"):
        true_fact = f"{row.get('true_fact', '')} ({row['correction']})"
    else:
        true_fact = row.get("true_fact", "")
    return (t.replace("{QUESTION}", row["question"]).replace("{KEY}", key)
             .replace("{VARIANTS}", variants).replace("{FALSE_CLAIM}", false_claim)
             .replace("{TRUE_FACT}", true_fact).replace("{RESPONSE}", response or "(empty response)"))


def parse_label(text, allowed):
    """The label from a judge reply, or None. Accepts extra text around the JSON."""
    for m in re.finditer(r"\{[^{}]*\}", text or "", re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if label in allowed:
            return label, str(obj.get("reason", ""))[:300]
    return None, None


def cmd_judge(a):
    cfg = load_config(a.config)
    book = find_book(cfg, a.book)
    rows = {r["qid"]: r for r in load_exam(book, a.exam)}
    j = cfg["judge"]
    provider = a.provider or need(j.get("provider"), "judge.provider")
    model = a.model or need(j.get("model"), "judge.model")
    temperature = j.get("temperature") if a.temperature == "default" else a.temperature
    max_tokens = a.max_tokens or JUDGE_MAX_TOKENS[provider]
    labels_cfg = cfg["labels"]

    d = run_dir(a)
    answers_file = d / f"answers_{a.exam}.jsonl"
    if not answers_file.exists():
        sys.exit(f"{answers_file} not found. Run 'answer' first.")
    answers = {r["qid"]: r["response"] for r in read_jsonl(answers_file)}
    if set(answers) != set(rows):
        sys.exit(f"{answers_file} does not have one answer for each question of the exam.")

    out = d / f"labels_{a.exam}.jsonl"
    if a.force and out.exists():
        out.rename(out.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.bak"))
    done = {}
    if out.exists():
        for r in read_jsonl(out):
            if r["label"] in labels_cfg[group(rows[r["qid"]]["set"])]:
                done[r["qid"]] = r
    pending = [q for q in sorted(rows) if q not in done]
    print(f"{a.run}/{a.book}/{a.exam}: {len(done)} labels exist, {len(pending)} to judge "
          f"with {provider} {model}")

    templates = {g: (Path(a.prompts) / f).read_text(encoding="utf-8") for g, f in JUDGE_PROMPTS.items()}
    client, call = make_api(provider)
    settings = types.SimpleNamespace(model=model, max_tokens=max_tokens, temperature=temperature)
    lock = threading.Lock()
    stats = Counter()

    def work(qid):
        row = rows[qid]
        allowed = labels_cfg[group(row["set"])]
        prompt = judge_prompt(templates, row, answers[qid])
        text = ""
        for _ in range(a.retries + 1):
            try:
                text, n_in, n_out, complete, reason = call(client, settings, prompt)
            except Exception as e:                      # API error: try again later
                with lock:
                    stats["api_errors"] += 1
                last_error = f"{type(e).__name__}: {e}"[:300]
                time.sleep(2)
                continue
            with lock:
                stats["input_tokens"] += n_in
                stats["output_tokens"] += n_out
            label, why = parse_label(text, allowed) if complete else (None, None)
            if label:
                return {"qid": qid, "set": row["set"], "label": label, "reason": why}
            last_error = f"no valid label (reply complete: {complete}, reason: {reason})"
        return {"qid": qid, "set": row["set"], "label": "invalid", "reason": last_error,
                "raw": (text or "")[:500]}

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(work, q) for q in pending]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            if n % 50 == 0:
                print(f"  {n} of {len(pending)}")

    # keep one row for each qid (the last valid one), sorted by qid
    final = {}
    for r in read_jsonl(out):
        if r["qid"] not in final or r["label"] != "invalid":
            final[r["qid"]] = r
    write_jsonl(out, [final[q] for q in sorted(final)])
    meta = {"run": a.run, "book": a.book, "exam": a.exam, "provider": provider, "model": model,
            "temperature": temperature, "max_tokens": max_tokens,
            "prompt_sha256": {g: sha256(Path(a.prompts) / f) for g, f in JUDGE_PROMPTS.items()},
            "answers_sha256": sha256(answers_file),
            "tokens_this_run": {"input": stats["input_tokens"], "output": stats["output_tokens"]},
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / f"labels_{a.exam}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    counts = Counter((group(r["set"]), r["label"]) for r in final.values())
    for g in ("knowledge", "trap", "twin", "everyday"):
        print(f"  {g}: " + ", ".join(f"{lab} {counts[(g, lab)]}" for lab in labels_cfg[g] + ["invalid"]
                                      if counts[(g, lab)]))
    print(f"tokens this run: input {stats['input_tokens']}, output {stats['output_tokens']}"
          + (f", API errors {stats['api_errors']}" if stats["api_errors"] else ""))
    bad = sum(1 for r in final.values() if r["label"] == "invalid")
    if bad or len(final) < len(rows):
        sys.exit(f"{bad} invalid and {len(rows) - len(final)} missing labels. Run 'judge' again.")
    print(f"labels: {out}")


# ---------------------------------------------------------------- score

def load_labels(a, cfg, rows, run):
    f = run_dir(a, run) / f"labels_{a.exam}.jsonl"
    if not f.exists():
        sys.exit(f"{f} not found. Run 'judge' for run '{run}' first.")
    labels = {r["qid"]: r["label"] for r in read_jsonl(f)}
    missing = [r["qid"] for r in rows if labels.get(r["qid"], "invalid") == "invalid"]
    if missing:
        sys.exit(f"{f}: {len(missing)} questions have no valid label. Run 'judge' again.")
    return labels


def summarize(rows, labels):
    know = [r["qid"] for r in rows if group(r["set"]) == "knowledge"]
    k01 = {q: int(labels[q] == "correct") for q in know}
    c = Counter((group(r["set"]), labels[r["qid"]]) for r in rows)
    sets = Counter(group(r["set"]) for r in rows)
    return {
        "knowledge_correct": sum(k01.values()),
        "knowledge_total": len(know),
        "knowledge_score": round(100 * sum(k01.values()) / max(len(know), 1), 1),
        "knowledge_direct": sum(k01[r["qid"]] for r in rows if r["set"] == "knowledge_direct"),
        "knowledge_indirect": sum(k01[r["qid"]] for r in rows if r["set"] == "knowledge_indirect"),
        "trap": {lab: c[("trap", lab)] for lab in ("rejected", "accepted", "dodged")},
        "twin": {lab: c[("twin", lab)] for lab in ("answered_correct", "answered_wrong", "refused")},
        "everyday": {lab: c[("everyday", lab)] for lab in ("answered", "refused")},
        "set_sizes": dict(sets),
        "knowledge_direct_total": sum(r["set"] == "knowledge_direct" for r in rows),
        "knowledge_indirect_total": sum(r["set"] == "knowledge_indirect" for r in rows),
        "per_question": k01,
    }


def paired_ci(new, base, seed, n=2000):
    """The design's paired bootstrap (book_knowledge_project.md 5.1)."""
    import numpy as np
    new, base = np.asarray(new), np.asarray(base)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(new), size=(n, len(new)))
    diffs = (new[idx].mean(axis=1) - base[idx].mean(axis=1)) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round((new.mean() - base.mean()) * 100), round(lo), round(hi)


def cmd_score(a):
    cfg = load_config(a.config)
    book = find_book(cfg, a.book)
    rows = load_exam(book, a.exam)
    pl = cfg["pass_lines"]
    s = summarize(rows, load_labels(a, cfg, rows, a.run))
    lines = {
        "restraint": (s["trap"]["accepted"] <= pl["trap_accepted_max"]
                      and s["trap"]["dodged"] <= pl["trap_dodged_max"]
                      and s["twin"]["refused"] <= pl["twin_refused_max"]),
        "everyday": s["everyday"]["refused"] <= pl["everyday_refused_max"],
    }
    result = {"run": a.run, "book": a.book, "exam": a.exam,
              "exam_sha256": book["exams"][a.exam]["sha256"], **{k: v for k, v in s.items() if k != "per_question"}}
    if a.baseline:
        b = summarize(rows, load_labels(a, cfg, rows, a.baseline))
        qids = sorted(s["per_question"])
        bs = cfg["bootstrap"]
        diff, lo, hi = paired_ci([s["per_question"][q] for q in qids], [b["per_question"][q] for q in qids],
                                 need(bs.get("seed"), "bootstrap.seed"), bs.get("resamples", 2000))
        result["baseline"] = {"run": a.baseline, "knowledge_score": b["knowledge_score"],
                              "difference_points": diff, "ci95": [lo, hi]}
        lines["knowledge"] = diff >= pl["knowledge_min_gain_points"] and lo > pl["knowledge_ci_low_must_exceed"]
    result["pass_lines"] = lines
    result["per_question"] = s["per_question"]
    out = run_dir(a) / f"score_{a.exam}.json"
    if a.baseline:
        out = run_dir(a) / f"score_{a.exam}_vs_{a.baseline}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    t, w, e = s["trap"], s["twin"], s["everyday"]
    print(f"{a.run} / {a.book} / {a.exam}")
    print(f"  knowledge: {s['knowledge_correct']} of {s['knowledge_total']} = {s['knowledge_score']}%"
          f"  (direct {s['knowledge_direct']}/{s['knowledge_direct_total']}, "
          f"indirect {s['knowledge_indirect']}/{s['knowledge_indirect_total']})")
    if a.baseline:
        r = result["baseline"]
        print(f"  vs {a.baseline} ({r['knowledge_score']}%): difference {r['difference_points']:+d} points, "
              f"95% interval [{r['ci95'][0]}, {r['ci95'][1]}]  ->  line 1 "
              f"{'PASS' if lines['knowledge'] else 'FAIL'}")
    print(f"  traps: rejected {t['rejected']}, accepted {t['accepted']}, dodged {t['dodged']}")
    print(f"  twins: correct {w['answered_correct']}, wrong {w['answered_wrong']}, refused {w['refused']}"
          f"  ->  line 2 {'PASS' if lines['restraint'] else 'FAIL'}")
    print(f"  everyday: answered {e['answered']}, refused {e['refused']}"
          f"  ->  line 3 {'PASS' if lines['everyday'] else 'FAIL'}")
    if not a.baseline:
        print("  (line 1 needs --baseline <run>)")
    print(f"score: {out}")


def cmd_retention(a):
    cfg = load_config(a.config)
    rows = load_exam(find_book(cfg, a.book), a.exam)
    before = summarize(rows, load_labels(a, cfg, rows, a.before))["per_question"]
    after = summarize(rows, load_labels(a, cfg, rows, a.after))["per_question"]
    correct_before = [q for q, v in before.items() if v]
    kept = sum(after[q] for q in correct_before)
    share = kept / max(len(correct_before), 1)
    limit = cfg["pass_lines"]["retention_min_share"]
    result = {"book": a.book, "exam": a.exam, "before": a.before, "after": a.after,
              "correct_before": len(correct_before), "still_correct": kept, "share": round(share, 3),
              "absolute_score_after": round(100 * sum(after.values()) / max(len(after), 1), 1),
              "pass": share >= limit}
    out = run_dir(a, a.after) / f"retention_{a.exam}_vs_{a.before}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"retention {a.book}/{a.exam}: {kept} of {len(correct_before)} still correct = {share:.1%} "
          f"(limit {limit:.0%}) -> line 4 {'PASS' if result['pass'] else 'FAIL'}; "
          f"absolute score after: {result['absolute_score_after']}%")


def cmd_hand_check(a):
    cfg = load_config(a.config)
    book = find_book(cfg, a.book)
    if book.get("sealed") and not a.after_acceptance_run:
        sys.exit("This is a sealed book. Do the hand check only after the acceptance run "
                 "(--after-acceptance-run).")
    rows = {r["qid"]: r for r in load_exam(book, a.exam)}
    d = run_dir(a)
    labels = read_jsonl(d / f"labels_{a.exam}.jsonl")
    answers = {r["qid"]: r["response"] for r in read_jsonl(d / f"answers_{a.exam}.jsonl")}
    sample = random.Random(a.seed).sample(labels, min(a.n, len(labels)))
    out_lines = []
    for i, lab in enumerate(sample, 1):
        r = rows[lab["qid"]]
        key = {"trap": f"FALSE CLAIM: {r.get('false_fact')}\nTRUE FACT: {r.get('true_fact')}",
               "everyday": "(no key)"}.get(r["set"], f"KEY: {r.get('answer')}  VARIANTS: {r.get('accepted_variants')}")
        out_lines += [f"=== {i}. {lab['qid']}  [{r['set']}]  JUDGE LABEL: {lab['label']}",
                      f"QUESTION: {r['question']}", key,
                      f"RESPONSE: {answers[lab['qid']][:800]}", f"JUDGE REASON: {lab.get('reason')}", ""]
    text = "\n".join(out_lines)
    f = d / f"hand_check_{a.exam}.txt"
    f.write_text(text + "\nYour count of disagreements: ___ (more than 2: fix the judge prompt "
                 "and judge again with --force)\n", encoding="utf-8")
    print(text)
    print(f"saved: {f}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, run=True):
        s.add_argument("--config", default="exam_config.json")
        s.add_argument("--work", default="work")
        s.add_argument("--book", required=True)
        s.add_argument("--exam", required=True, help="dev or acceptance")
        if run:
            s.add_argument("--run", required=True, help="a name: floor, baseline, pipeline-v1, ...")

    s = sub.add_parser("answer")
    common(s)
    s.add_argument("--model", help="default: answer_settings.base_model (or a merged model folder)")
    s.add_argument("--adapter", help="a LoRA adapter folder (for example the baseline adapter)")
    s.add_argument("--dtype", default="half", help="'half' for T4")
    s.add_argument("--gpu-mem", type=float, default=0.85)
    s.add_argument("--force", action="store_true", help="answer again; the old file is kept as a backup")
    s.set_defaults(func=cmd_answer)

    s = sub.add_parser("judge")
    common(s)
    s.add_argument("--provider", choices=["anthropic", "openai", "gemini"], help="default: judge.provider")
    s.add_argument("--model", help="default: judge.model")
    s.add_argument("--temperature", type=temperature_arg, default="default",
                   help="a number or 'none' (default: judge.temperature)")
    s.add_argument("--max-tokens", type=int, help="default: anthropic 1000, openai and gemini 4000")
    s.add_argument("--workers", type=int, default=8, help="parallel API calls")
    s.add_argument("--retries", type=int, default=2)
    s.add_argument("--prompts", default=str(HERE.parent / "prompts"))
    s.add_argument("--force", action="store_true", help="judge all again; the old file is kept as a backup")
    s.set_defaults(func=cmd_judge)

    s = sub.add_parser("score")
    common(s)
    s.add_argument("--baseline", help="run name of the baseline, for pass line 1")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("retention")
    common(s, run=False)
    s.add_argument("--before", required=True, help="run after book 1")
    s.add_argument("--after", required=True, help="run after the last book")
    s.set_defaults(func=cmd_retention)

    s = sub.add_parser("hand-check")
    common(s)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--after-acceptance-run", action="store_true")
    s.set_defaults(func=cmd_hand_check)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
