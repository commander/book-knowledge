#!/usr/bin/env python3
"""The fixed baseline recipe (book_knowledge_project.md 4.1), lab side.

All recipe values come from exam_config.json ("baseline_recipe"). Do not change
them: the baseline score is the reference for pass line 1.

Two commands. Run them as separate commands (the teacher and the student must
not be on a T4 at the same time):

  teacher  Qwen2.5-7B-Instruct-AWQ (vLLM) writes question-answer pairs for each
           unit of the book.
  train    QLoRA on the pairs, from the base model, this book only. Saves the
           LoRA adapter. Continues from the last checkpoint after a disconnect.

  python src/baseline_recipe.py teacher --book critical-software-studies
  python src/baseline_recipe.py train   --book critical-software-studies
  python src/lab_eval.py answer --book critical-software-studies --exam dev --run baseline \\
         --adapter work/runs/baseline/critical-software-studies/adapter

Output in work/runs/<run>/<book>/ (run "baseline" by default):
  teacher_pairs.jsonl   {unit_id, question, answer}
  teacher_raw.jsonl     the full teacher reply for each unit
  teacher.meta.json     prompt, settings, counts, GPU, time
  checkpoints/          training checkpoints (deleted after a complete run with --clean)
  adapter/              the LoRA adapter
  train.meta.json       settings, steps, losses, GPU, time, adapter hash
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exam_builder import read_jsonl, write_jsonl  # noqa: E402

# The recipe gives no output form. This fixed line makes the reply readable by
# parse_pairs(). The full prompt is saved in teacher.meta.json.
FORMAT_LINE = ("Write each pair on two lines, in this form:\n"
               "Q: <question>\n"
               "A: <answer>")
Q_LINE = re.compile(r"^\W*(?:\d+\s*[.)])?\W*(?:Q|Question)\s*\d*\s*[:.)]\s*(.*)$", re.I)
A_LINE = re.compile(r"^\W*(?:A|Answer)\s*\d*\s*[:.)]\s*(.*)$", re.I)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"{p} not found. Make it with src/make_exam_config.py.")
    return json.loads(p.read_text(encoding="utf-8"))


def find_book(cfg, name):
    for b in cfg["books"]:
        if b["name"] == name:
            return b
    sys.exit(f"book '{name}' is not in exam_config.json")


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    except Exception:
        return "unknown"


def versions(*names):
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def clean(s):
    return re.sub(r"\s+", " ", s.replace("**", "").replace("__", "")).strip(" *_")


def parse_pairs(text):
    """Question-answer pairs from a teacher reply ("Q: ... / A: ..." and similar forms)."""
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


def out_dir(a):
    d = Path(a.work) / "runs" / a.run / a.book
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_units(a, book):
    path = Path(a.work) / a.book / "units.jsonl"
    if book.get("units_sha256") and sha256(path) != book["units_sha256"]:
        sys.exit(f"{path} does not match units_sha256 in exam_config.json")
    return read_jsonl(path)


# ---------------------------------------------------------------- teacher

def cmd_teacher(a):
    cfg = load_config(a.config)
    r = cfg["baseline_recipe"]
    book = find_book(cfg, a.book)
    units = load_units(a, book)
    d = out_dir(a)
    if (d / "teacher_pairs.jsonl").exists() and not a.force:
        sys.exit(f"{d / 'teacher_pairs.jsonl'} exists. Use --force to write it again.")

    start = time.time()
    from vllm import LLM, SamplingParams  # needs a GPU
    llm = LLM(model=r["teacher"], dtype="half", max_model_len=2048,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    tok = llm.get_tokenizer()
    instruction = f"{r['teacher_prompt']}\n\n{FORMAT_LINE}\n\nPassage:\n"
    prompts = [tok.apply_chat_template([{"role": "user", "content": instruction + u["text"]}],
                                       tokenize=False, add_generation_prompt=True) for u in units]
    params = SamplingParams(temperature=r["teacher_temperature"], max_tokens=a.max_tokens, seed=a.seed)
    outs = llm.generate(prompts, params)

    pairs, raw, short = [], [], 0
    n = r["pairs_per_unit"]
    for u, o in zip(units, outs):
        text = o.outputs[0].text
        got = parse_pairs(text)[:n]
        short += len(got) < n
        raw.append({"unit_id": u["id"], "reply": text, "pairs_found": len(got)})
        pairs += [{"unit_id": u["id"], "question": q, "answer": ans} for q, ans in got]
    write_jsonl(d / "teacher_pairs.jsonl", pairs)
    write_jsonl(d / "teacher_raw.jsonl", raw)
    meta = {"book": a.book, "run": a.run, "teacher": r["teacher"],
            "prompt_template": instruction + "<unit text>",
            "temperature": r["teacher_temperature"], "max_tokens": a.max_tokens, "seed": a.seed,
            "units": len(units), "pairs": len(pairs), "units_with_fewer_pairs": short,
            "gpu": gpu_name(), "versions": versions("vllm", "torch", "transformers"),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - start, 1)}
    (d / "teacher.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{a.book}: {len(pairs)} pairs from {len(units)} units "
          f"({short} units with fewer than {n} pairs) in {meta['seconds']} s on {meta['gpu']}")
    print(f"pairs: {d / 'teacher_pairs.jsonl'}")


# ---------------------------------------------------------------- train

def build_examples(tok, pairs, max_len):
    """Token ids and labels for each pair. Only the answer tokens are trained.

    The prompt is the default chat template with only the user turn (the same
    form as in the exam). The answer tokens follow the prompt tokens, as at
    answer time.
    """
    examples, cut = [], 0
    for p in pairs:
        prompt = tok.apply_chat_template([{"role": "user", "content": p["question"]}],
                                         tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template([{"role": "user", "content": p["question"]},
                                        {"role": "assistant", "content": p["answer"]}], tokenize=False)
        if not full.startswith(prompt):
            raise ValueError("the chat template does not start the full text with the prompt")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        cut += len(p_ids) + len(a_ids) > max_len
        if all(x == -100 for x in labels):
            continue                         # no answer token left after the cut
        examples.append({"input_ids": ids, "labels": labels})
    return examples, cut


def collate(pad_id):
    import torch

    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}
    return fn


def cmd_train(a):
    cfg = load_config(a.config)
    r = cfg["baseline_recipe"]
    base = a.base_model or cfg["answer_settings"]["base_model"]
    d = out_dir(a)
    pairs_file = d / "teacher_pairs.jsonl"
    if not pairs_file.exists():
        sys.exit(f"{pairs_file} not found. Run 'teacher' first.")
    if (d / "adapter" / "adapter_model.safetensors").exists() and not a.force:
        sys.exit(f"{d / 'adapter'} exists. Use --force to train again.")
    eff = r["effective_batch_size"]
    if eff % a.per_device_batch:
        sys.exit(f"effective_batch_size {eff} is not a multiple of --per-device-batch {a.per_device_batch}")
    if r.get("precision") != "fp16":
        sys.exit("this script trains in fp16 (T4); baseline_recipe.precision must be 'fp16'")

    start = time.time()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(base)
    pairs = read_jsonl(pairs_file)
    examples, cut = build_examples(tok, pairs, a.max_len)
    print(f"{len(examples)} training examples ({cut} cut to {a.max_len} tokens)")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.float16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=r["lora_rank"], lora_alpha=r["lora_alpha"], target_modules=r["target_modules"],
                      lora_dropout=a.lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    ckpt = d / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=r["epochs"], learning_rate=r["learning_rate"],
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=eff // a.per_device_batch,
        fp16=True, logging_steps=10, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=2, seed=a.seed, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=examples,
                      data_collator=collate(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id))
    resume = any(ckpt.glob("checkpoint-*"))
    if resume:
        print(f"continuing from the last checkpoint in {ckpt}")
    trainer.train(resume_from_checkpoint=True if resume else None)

    adapter = d / "adapter"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    meta = {"book": a.book, "run": a.run, "base_model": base, "recipe": r,
            "per_device_batch": a.per_device_batch, "gradient_accumulation": eff // a.per_device_batch,
            "lora_dropout": a.lora_dropout, "max_len": a.max_len, "seed": a.seed,
            "lr_scheduler": "linear (Trainer default), no warmup", "optimizer": "adamw_torch (Trainer default)",
            "quantization": "4-bit nf4, double quant, fp16 compute",
            "examples": len(examples), "examples_cut": cut, "steps": trainer.state.global_step,
            "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
            "resumed": resume, "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
            "pairs_sha256": sha256(pairs_file), "gpu": gpu_name(),
            "versions": versions("torch", "transformers", "peft", "bitsandbytes", "accelerate"),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds_this_session": round(time.time() - start, 1)}
    (d / "train.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{a.book}: {meta['steps']} steps, loss {meta['first_loss']} -> {meta['last_loss']}, "
          f"{meta['seconds_this_session']} s on {meta['gpu']}")
    print(f"adapter: {adapter}")
    if a.clean:
        import shutil
        shutil.rmtree(ckpt, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--config", default="exam_config.json")
        s.add_argument("--work", default="work")
        s.add_argument("--book", required=True)
        s.add_argument("--run", default="baseline")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--force", action="store_true")

    s = sub.add_parser("teacher")
    common(s)
    s.add_argument("--gpu-mem", type=float, default=0.9)
    s.add_argument("--max-tokens", type=int, default=1024, help="teacher reply limit for each unit")
    s.set_defaults(func=cmd_teacher)

    s = sub.add_parser("train")
    common(s)
    s.add_argument("--base-model", help="default: answer_settings.base_model")
    s.add_argument("--per-device-batch", type=int, default=4,
                   help="micro-batch; gradient accumulation makes the effective batch")
    s.add_argument("--max-len", type=int, default=512)
    s.add_argument("--lora-dropout", type=float, default=0.05, help="not in the recipe; recorded")
    s.add_argument("--save-steps", type=int, default=50)
    s.add_argument("--clean", action="store_true", help="delete the checkpoints after a complete run")
    s.set_defaults(func=cmd_train)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
