"""Shared helpers: files, hashes, the run context, GPU name, package versions."""
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "pipeline_prompts"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Write to a temporary file, then rename: a disconnect cannot leave half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def code_hashes():
    """SHA-256 of each pipeline file (code, prompts). Recorded for the freeze."""
    files = [REPO / "src" / "run_book.py", *sorted((REPO / "src" / "book_pipeline").glob("*.py")),
             *sorted(PROMPTS.glob("*.txt"))]
    return {str(f.relative_to(REPO)): sha256_file(f) for f in files if f.exists()}


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "no GPU"
    except Exception:
        return "no GPU"


def versions(*names):
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


@dataclass
class Ctx:
    """What a stage gets: the config, the book text file and the folders."""
    cfg: dict
    book_path: Path
    state: Path
    book_id: str

    @property
    def book_dir(self):
        return self.state / self.book_id

    @property
    def model_dir(self):
        return self.state / "model"

    @property
    def seed(self):
        return self.cfg["seed"]

    def start_model(self):
        """The model that this book starts from: the merged model of the last book, or the base model."""
        return str(self.model_dir) if (self.model_dir / "config.json").exists() else self.cfg["student"]
