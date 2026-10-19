#!/usr/bin/env python3
"""Prepare all books in a config file: source (EPUB or PDF) -> clean text -> units.

  python src/prepare_books.py --config books.json
  python src/prepare_books.py --config books.json --only critical-software-studies

For each book in the config, in <work_dir>/<name>/:
  book.txt          clean text (epub_to_text.py or pdf_to_text.py)
  book.report.txt   removal counts, leftovers, 20 random paragraphs
  units.jsonl       units for exam_builder.py (exam_builder.py split)
  convert.log       full output of the converter and of split
  prepare.json      record of the run: source hash, options, script hashes,
                    package versions, word and unit counts
and <work_dir>/prepare_summary.tsv with one row per book.

The same config and the same source files give the same output. To change a
book, change the config (or the file in books_dir) and run again.

Config (paths are relative to the config file):
{
  "books_dir": "books",
  "work_dir": "work",
  "split": {"min_words": 150, "max_words": 250},
  "books": [
    {"name": "book-a", "title": "Book A", "source": "book-a.epub", "options": {}},
    {"name": "book-b", "title": "Book B", "source": "book-b.pdf",
     "options": {"first_page": 18, "last_page": 247, "strip_author_year": true}},
    {"name": "sealed-2", "source": "sealed/x.epub", "sealed": true, "options": {}}
  ]
}
"options" are the converter options without "--" ("first_page" -> --first-page).
"sealed": true writes reports with counts only and prints only counts.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from text_check import find_leftovers, paragraphs  # noqa: E402

CONVERTERS = {".epub": "epub_to_text.py", ".pdf": "pdf_to_text.py"}
SCRIPTS = ["epub_to_text.py", "pdf_to_text.py", "text_check.py", "exam_builder.py", "prepare_books.py"]
PACKAGES = ["pymupdf", "beautifulsoup4", "lxml"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def option_args(options):
    args = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            args += [flag] + [str(v) for v in value]
        else:
            args += [flag, str(value)]
    return args


def run(cmd, log):
    log.write("$ " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write(r.stdout + r.stderr + "\n")
    log.flush()
    return r


def versions():
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def prepare(book, cfg, base, a):
    name = book["name"]
    src = base / cfg.get("books_dir", "books") / book["source"]
    out_dir = base / cfg.get("work_dir", "work") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed = bool(book.get("sealed"))
    row = {"name": name, "status": "failed", "words": "", "units": "", "leftovers": ""}

    if not src.exists():
        print(f"{name}: source file not found: {src}")
        return row
    converter = CONVERTERS.get(src.suffix.lower())
    if not converter:
        print(f"{name}: unknown source type {src.suffix} (use .epub or .pdf)")
        return row

    txt = out_dir / "book.txt"
    opts = option_args(book.get("options", {})) + (["--counts-only"] if sealed else [])
    split_cfg = cfg.get("split", {})
    start = time.time()
    with open(out_dir / "convert.log", "w", encoding="utf-8") as log:
        r = run([sys.executable, str(HERE / converter), "convert", str(src), str(txt)] + opts, log)
        if r.returncode != 0:
            print(f"{name}: conversion failed, see {out_dir / 'convert.log'}")
            if not sealed:
                print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
            return row
        r = run([sys.executable, str(HERE / "exam_builder.py"), "split", "--dir", str(out_dir),
                 "--book", str(txt),
                 "--min-words", str(split_cfg.get("min_words", 150)),
                 "--max-words", str(split_cfg.get("max_words", 250))], log)
        if r.returncode != 0:
            print(f"{name}: split failed, see {out_dir / 'convert.log'}")
            return row

    paras = paragraphs(txt.read_text(encoding="utf-8"))
    words = sum(len(p.split()) for p in paras)
    units = sum(1 for line in open(out_dir / "units.jsonl", encoding="utf-8") if line.strip())
    leftovers = {k: c for k, (c, _) in find_leftovers(paras, show=0).items()}
    record = {
        "name": name,
        "title": book.get("title"),
        "role": book.get("role"),
        "sealed": sealed,
        "source": book["source"],
        "source_sha256": sha256(src),
        "converter": converter,
        "options": book.get("options", {}),
        "split": {"min_words": split_cfg.get("min_words", 150), "max_words": split_cfg.get("max_words", 250)},
        "words": words,
        "units": units,
        "leftovers": leftovers,
        "book_txt_sha256": sha256(txt),
        "units_sha256": sha256(out_dir / "units.jsonl"),
        "script_sha256": {s: sha256(HERE / s) for s in SCRIPTS if (HERE / s).exists()},
        "versions": versions(),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - start, 1),
    }
    (out_dir / "prepare.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    row.update(status="ok", words=words, units=units,
               leftovers=", ".join(f"{k} {v}" for k, v in leftovers.items()) or "none")
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="*", help="prepare only these book names")
    a = p.parse_args()

    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    books = [b for b in cfg["books"] if not a.only or b["name"] in a.only]
    if a.only and len(books) != len(a.only):
        sys.exit(f"unknown name in --only. Names in the config: {[b['name'] for b in cfg['books']]}")
    names = [b["name"] for b in cfg["books"]]
    if len(set(names)) != len(names):
        sys.exit("each book needs a different name")

    rows = []
    for book in books:
        print(f"--- {book['name']}")
        rows.append(prepare(book, cfg, base, a))

    print()
    print(f"{'book':<45} {'status':<7} {'words':>7} {'units':>6}  leftovers")
    for r in rows:
        print(f"{r['name']:<45} {r['status']:<7} {r['words']!s:>7} {r['units']!s:>6}  {r['leftovers']}")
    summary = base / cfg.get("work_dir", "work") / "prepare_summary.tsv"
    if not a.only:
        with open(summary, "w", encoding="utf-8") as f:
            f.write("book\tstatus\twords\tunits\tleftovers\n")
            for r in rows:
                f.write(f"{r['name']}\t{r['status']}\t{r['words']}\t{r['units']}\t{r['leftovers']}\n")
        print(f"\nsummary: {summary}")
    if any(r["status"] != "ok" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
