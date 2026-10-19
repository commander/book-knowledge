"""Stage 7: merge the adapter into the start model and save it as state/model.

Order, so that a disconnect at any point can be repaired by running the stage again:
  1. merge into state/model.tmp
  2. move the old state/model to state/model.prev (if it exists)
  3. move state/model.tmp to state/model, write <book>/merged.ok
  4. save the replay sample, remove the checkpoints and state/model.prev
"""
import random
import shutil
import time

from .common import read_json, read_jsonl, write_json, write_jsonl


def run(ctx):
    d = ctx.book_dir / "train"
    start_model = read_json(d / "start_model.json")["start_model"]
    adapter = d / "adapter"
    model_dir, tmp, prev = ctx.model_dir, ctx.state / "model.tmp", ctx.state / "model.prev"
    if start_model == str(model_dir) and prev.exists():
        start_model = str(prev)              # a restart after step 2

    start = time.time()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_ok = ctx.book_dir / "merged.ok"   # state/model is already the merged model: do not merge twice
    if not merged_ok.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        model = AutoModelForCausalLM.from_pretrained(start_model, torch_dtype=torch.float16, device_map={"": 0})
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="2GB")
        AutoTokenizer.from_pretrained(start_model).save_pretrained(str(tmp))
        del model
        torch.cuda.empty_cache()

        if model_dir.exists() and not prev.exists():
            model_dir.rename(prev)
        shutil.rmtree(model_dir, ignore_errors=True)
        tmp.rename(model_dir)
        merged_ok.write_text(start_model, encoding="utf-8")

    rows = []
    for f in sorted((ctx.book_dir / "data").glob("*.jsonl")):
        rows += read_jsonl(f)
    n = min(ctx.cfg["replay"]["save_rows"], len(rows))
    write_jsonl(ctx.state / "replay" / f"{ctx.book_id}.jsonl", random.Random(ctx.seed).sample(rows, n))

    shutil.rmtree(d / "checkpoints", ignore_errors=True)
    shutil.rmtree(prev, ignore_errors=True)
    write_json(ctx.book_dir / "save.meta.json",
               {"merged_from": start_model, "adapter": str(adapter), "replay_rows_saved": n,
                "seconds": round(time.time() - start, 1)})
    print(f"save: merged model in {model_dir}, {n} replay rows saved, {round(time.time() - start)} s")
