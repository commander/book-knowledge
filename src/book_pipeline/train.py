"""Stage 5: QLoRA training of the student on this book's data, plus replay.

The start model is state/model (the merged model of the last book), or the base
model for the first book. The start model is recorded before the training, so a
restart after a disconnect continues from the same model and checkpoint.
"""
import math
import random
import time

from .common import gpu_name, read_json, read_jsonl, sha256_file, versions, write_json


def load_rows(ctx):
    rows = []
    for f in sorted((ctx.book_dir / "data").glob("*.jsonl")):
        rows += read_jsonl(f)
    return rows


def replay_rows(ctx, n_new):
    """share x n_new rows from the earlier books, in equal parts for each book."""
    share = ctx.cfg["replay"]["share"]
    files = [f for f in sorted((ctx.state / "replay").glob("*.jsonl")) if f.stem != ctx.book_id]
    if not share or not files:
        return [], {}
    per_book = int(share * n_new / len(files))
    rng = random.Random(ctx.seed)
    out, counts = [], {}
    for f in files:
        rows = read_jsonl(f)
        pick = rng.sample(rows, min(per_book, len(rows)))
        out += pick
        counts[f.stem] = len(pick)
    return out, counts


def build_examples(tok, rows, max_len):
    """Token ids and labels. Only the answer tokens are trained.

    The prompt is the default chat template with one user turn (the same form as
    the exam). The answer tokens follow the prompt tokens, as at answer time.
    """
    examples, cut = [], 0
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                         tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template([{"role": "user", "content": r["question"]},
                                        {"role": "assistant", "content": r["answer"]}], tokenize=False)
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


def run(ctx):
    t = ctx.cfg["train"]
    d = ctx.book_dir / "train"
    d.mkdir(parents=True, exist_ok=True)
    eff = t["effective_batch_size"]
    if eff % t["per_device_batch"]:
        raise SystemExit("train.effective_batch_size is not a multiple of train.per_device_batch")

    start_file = d / "start_model.json"
    start_model = (read_json(start_file) or {}).get("start_model")
    if start_model is None:
        start_model = ctx.start_model()
        write_json(start_file, {"start_model": start_model})

    start = time.time()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(start_model)
    rows = load_rows(ctx)
    replay, replay_counts = replay_rows(ctx, len(rows))
    random.Random(ctx.seed).shuffle(all_rows := rows + replay)
    examples, cut = build_examples(tok, all_rows, t["max_len"])
    print(f"train: {len(rows)} rows from this book, {len(replay)} replay rows, "
          f"{len(examples)} examples ({cut} cut to {t['max_len']} tokens), start model {start_model}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(start_model, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.float16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=t["lora_rank"], lora_alpha=t["lora_alpha"], target_modules=t["target_modules"],
                      lora_dropout=t["lora_dropout"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    ckpt = d / "checkpoints"
    # Warmup as a number of steps: newer transformers versions do not accept warmup_ratio.
    total_steps = math.ceil(len(examples) / eff) * t["epochs"]
    warmup_steps = int(t["warmup_ratio"] * total_steps)
    args = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=t["epochs"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"], warmup_steps=warmup_steps,
        per_device_train_batch_size=t["per_device_batch"], gradient_accumulation_steps=eff // t["per_device_batch"],
        fp16=True, logging_steps=10, save_strategy="steps", save_steps=t["save_steps"],
        save_total_limit=2, seed=ctx.seed, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate(pad))
    resume = any(ckpt.glob("checkpoint-*"))
    if resume:
        print(f"train: continuing from the last checkpoint in {ckpt}")
    trainer.train(resume_from_checkpoint=True if resume else None)

    adapter = d / "adapter"
    model.save_pretrained(str(adapter))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    n_tokens = sum(len(e["input_ids"]) for e in examples)
    meta = {"start_model": start_model, "train": t, "rows_book": len(rows), "rows_replay": replay_counts,
            "examples": len(examples), "examples_cut": cut, "tokens_per_epoch": n_tokens,
            "steps": trainer.state.global_step, "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None, "loss_log": losses, "resumed": resume,
            "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"), "gpu": gpu_name(),
            "versions": versions("torch", "transformers", "peft", "bitsandbytes", "accelerate"),
            "seconds_this_session": round(time.time() - start, 1)}
    write_json(d / "train.meta.json", meta)
    print(f"train: {meta['steps']} steps, loss {meta['first_loss']} -> {meta['last_loss']}, "
          f"{n_tokens} tokens per epoch, {meta['seconds_this_session']} s on {meta['gpu']}")
