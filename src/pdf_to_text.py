#!/usr/bin/env python3
"""Convert a book PDF (with a text layer) to clean plain text for exam_builder.py.

Step 1 - inspect the PDF (find the body font size, headings, front and back matter):
  python pdf_to_text.py inspect book.pdf

Step 2 - convert the main text pages:
  python pdf_to_text.py convert book.pdf book.txt --first-page 15 --last-page 240

Step 3 - read book.report.txt (removed items, leftovers, 20 random paragraphs).

What "convert" removes:
  - running headers and footers (lines that repeat at the top or bottom of pages)
  - page numbers
  - footnotes and note markers (text smaller than the body font, superscripts)
  - figure, table and map captions; table-like blocks of numbers
  - rotated (landscape) pages, for example a wide table or figure printed
    sideways, and text lines that are not horizontal. This loses the content of
    these pages (see the report); use --keep-rotated to keep them (the reading
    order of a rotated page is then wrong)
  - control characters; a tab becomes a space
  - [12]-style citations, URLs, DOIs
  - chapter-start footers ("DOI: 10..." and "... CC BY license" lines)
  - "Notes", "References" and "Bibliography" sections: skipped until the next
    chapter heading (a heading at least --chapter-ratio x the body font size,
    or a "Chapter N" / "Part N" heading). This works for books with notes and
    references after each chapter.
  - everything from "Index" to the end
What it does:
  - joins lines into paragraphs, joins words split at a line end (soft
    hyphen, hyphen, or a zero-width break after "-" or "/"), joins paragraphs
    split across pages, normalizes ligatures (fi, fl) and quotes
  - joins a chapter title that wraps onto two lines
  - writes one paragraph per block, separated by a blank line

Requires: pip install pymupdf
Scanned PDFs (no text layer) need OCR first, for example: ocrmypdf in.pdf out.pdf
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from text_check import drop_citation_groups, write_report

BACK_STOP = re.compile(r"^(index|general index|index of names|subject index)$", re.I)
BACK_SKIP = re.compile(
    r"^(notes|endnotes|notes to chapter.*|chapter notes|references|bibliography|"
    r"select bibliography|works cited|sources|further reading)$", re.I)
CHAPTER_HEAD = re.compile(r"^(chapter|interchapter|part|appendix|conclusion|introduction)\b", re.I)
LICENCE_LINE = re.compile(r"^DOI:\s*10\.|\bCC[ -]BY\b.*licen[cs]e|made available under a", re.I)
SHY, ZWSP = "\u00ad", "\u200b"
CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f-\x9f]")
CAPTION = re.compile(r"^(figure|fig\.|table|plate|map|illustration|chart|source)\s*[\d.:IVX]", re.I)
SENT_END = re.compile(r"[.!?:;\"”’')\]]$")
HEAD_END = re.compile(r"[.,;:]$")          # a bold line that ends like this is not a heading
EDGE_NUMBER = re.compile(r"^(\d{1,4})\s|\s(\d{1,4})$|^(\d{1,4})$")


# ---------------------------------------------------------------- reading

def clean_chars(s):
    # soft hyphens and zero-width spaces stay until join_lines() uses them
    s = unicodedata.normalize("NFKC", s)
    s = CONTROL.sub("", s.replace("\t", " "))
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return s


def rotated_pages(doc, first, last):
    """PDF page numbers (1-based) of pages that are rotated (landscape)."""
    return [pno + 1 for pno in range(first - 1, last) if doc[pno].rotation % 360]


def read_lines(doc, first, last, keep_rotated=True, stats=None):
    """Return one record per text line: page, position, size, text, flags.

    keep_rotated=False: skip rotated pages, and lines that are not horizontal.
    """
    lines = []
    skip = set() if keep_rotated else set(rotated_pages(doc, first, last))
    for pno in range(first - 1, last):
        if pno + 1 in skip:
            if stats is not None:
                stats[f"rotated_page_skipped p.{pno + 1}"] += 1
            continue
        page = doc[pno]
        h, w = page.rect.height, page.rect.width
        d = page.get_text("dict", sort=True)
        for bi, block in enumerate(d["blocks"]):
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                if not keep_rotated and (round(line["dir"][0]), round(line["dir"][1])) != (1, 0):
                    if stats is not None:
                        stats["vertical_lines_removed"] += 1
                    continue
                lines.append({
                    "page": pno + 1, "block": (pno, bi),
                    "x0": line["bbox"][0], "y0": line["bbox"][1], "y1": line["bbox"][3],
                    "page_h": h, "page_w": w,
                    "spans": [{"text": clean_chars(s["text"]), "size": s["size"],
                               "flags": s["flags"]} for s in spans],
                })
    return lines


def body_size(lines):
    count = Counter()
    for ln in lines:
        for s in ln["spans"]:
            count[round(s["size"] * 2) / 2] += len(s["text"])
    return count.most_common(1)[0][0], count


def line_size(ln):
    c = Counter()
    for s in ln["spans"]:
        c[round(s["size"] * 2) / 2] += len(s["text"])
    return c.most_common(1)[0][0]


def is_bold(ln):
    return all(s["flags"] & 16 for s in ln["spans"])


def plain(ln):
    return "".join(s["text"] for s in ln["spans"]).strip()


def no_invisible(t):
    return re.sub("[\u00ad\u200b\u200c\u200d\ufeff]", "", t)


# ---------------------------------------------------------------- inspect

def cmd_inspect(a):
    doc = pymupdf.open(a.pdf)
    n = len(doc)
    chars = [len(doc[i].get_text()) for i in range(n)]
    print(f"pages: {n}   mean characters per page: {sum(chars) // max(n, 1)}")
    empty = sum(c < 200 for c in chars)
    if empty > n * 0.3:
        print("WARNING: many pages have almost no text. The PDF is probably scanned.")
        print("         Run OCR first (for example: ocrmypdf in.pdf out.pdf).")
    lines = read_lines(doc, 1, n)
    rot = rotated_pages(doc, 1, n)
    if rot:
        print(f"rotated (landscape) pages, skipped by convert: {rot}")
    body, hist = body_size(lines)
    total = sum(hist.values())
    print(f"body font size: {body}")
    print("font sizes (share of characters):",
          ", ".join(f"{s}: {c / total:.0%}" for s, c in sorted(hist.items()) if c / total >= 0.01))
    print("\nprobable headings (larger or bold short lines), with page numbers:")
    shown = 0
    for ln in lines:
        t = plain(ln)
        if len(t.split()) <= 10 and (line_size(ln) >= body * 1.15 or is_bold(ln)):
            if not re.fullmatch(r"[\divxlc]+", t, re.I):
                print(f"  p.{ln['page']:>4}  {t[:80]}")
                shown += 1
                if shown >= a.max_headings:
                    print("  ...")
                    break
    print("\nUse these to choose --first-page (first page of the main text, after the")
    print("contents and preface you do not want) and --last-page (last page before the")
    print("bibliography or index).")


# ---------------------------------------------------------------- convert

def header_footer_keys(lines, zone, min_repeats):
    counts = Counter()
    for ln in lines:
        rel = ln["y0"] / ln["page_h"]
        if rel < zone or ln["y1"] / ln["page_h"] > 1 - zone:
            counts[re.sub(r"\d+", "#", plain(ln).lower())] += 1
    return {k for k, c in counts.items() if c >= min_repeats}


def edge_number(t):
    m = EDGE_NUMBER.search(t.strip())
    return int(next(g for g in m.groups() if g)) if m else None


def page_label_offset(lines, zone):
    """Offset between PDF page and printed page number (PDF page - printed number).

    Found from numbers at the start or end of header/footer lines. None if
    there is no clear offset.
    """
    counts, pages = Counter(), set()
    for ln in lines:
        if ln["y0"] / ln["page_h"] < zone or ln["y1"] / ln["page_h"] > 1 - zone:
            n = edge_number(plain(ln))
            if n is not None:
                counts[ln["page"] - n] += 1
                pages.add(ln["page"])
    if not counts:
        return None
    off, c = counts.most_common(1)[0]
    return off if c >= max(3, 0.3 * len(pages)) else None


def is_running_head(ln, raw, offset, zone):
    """A short line near the top or bottom that holds this page's printed number."""
    if offset is None or len(raw.split()) > 12:
        return False
    near = ln["y0"] / ln["page_h"] < zone + 0.05 or ln["y1"] / ln["page_h"] > 1 - zone - 0.05
    return near and edge_number(raw) == ln["page"] - offset


def line_text(ln, body, a, stats):
    """Text of a line without footnote-size text and superscript markers."""
    out = []
    for s in ln["spans"]:
        small = s["size"] < body * a.small_ratio
        sup = bool(s["flags"] & 1)
        if small or (sup and re.fullmatch(r"[\s\d*†‡,\-]+", s["text"])):
            stats["small_or_superscript_spans"] += 1
            continue
        out.append(s["text"])
    return "".join(out).strip()


def join_lines(parts):
    text = ""
    for p in parts:
        if not text:
            text = p
        elif text.endswith(SHY):
            text = text[:-1] + p                          # hyphen\u00ad\nation -> hyphenation
        elif text.endswith(ZWSP):
            # break after "-", "/" or a dash: keep the character, no space,
            # but " –" (spaced dash) gets its space back
            text += (" " if re.search(r"\s[\u2013\u2014]" + ZWSP + "$", text) else "") + p
        elif re.search(r"[A-Za-z]-$", text) and re.match(r"[a-z]", p):
            text = text[:-1] + p                          # exam-\nple -> example
        else:
            text += " " + p
    return text


def clean_paragraph(t, a, stats):
    t = no_invisible(t)
    before = t
    t = re.sub(r"\s*[•●▪]\s*", " ", t).strip()                         # list bullets
    t = re.sub(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]", "", t)          # [12], [3-5]
    # drop the whole sentence that contains a URL or DOI
    t = re.sub(r"[^.!?]*(?:https?://|www\.|\bdoi:)\S*[^.!?]*[.!?]?\s*", "", t, flags=re.I)
    t = re.sub(r"(?<=[A-Za-z\]][.,;:\"')])\d{1,3}(?=\s|$)", "", t)        # end.12 marker (not 2.9)
    if a.strip_author_year:
        t = drop_citation_groups(t, stats)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+([.,;:!?)])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if t != before:
        stats["paragraphs_with_markers_removed"] += 1
    return t


def looks_like_table(t):
    toks = t.split()
    if len(toks) < 6:
        return False
    num = sum(bool(re.fullmatch(r"[\d.,%$£€()\-–]+", x)) for x in toks)
    return num / len(toks) > 0.4


def cmd_convert(a):
    doc = pymupdf.open(a.pdf)
    first = a.first_page or 1
    last = a.last_page or len(doc)
    stats = Counter()
    lines = read_lines(doc, first, last, keep_rotated=a.keep_rotated, stats=stats)
    if not lines:
        raise SystemExit("No text found. The PDF is probably scanned: run OCR first.")
    body, _ = body_size(lines)
    hf = header_footer_keys(lines, a.zone, a.min_repeats)
    offset = page_label_offset(lines, a.zone)

    # 1. lines -> raw paragraphs (with heading marks)
    paras = []          # dicts: text, heading, page
    cur, cur_block, prev_x0 = [], None, None

    def flush():
        if cur:
            paras.append({"text": join_lines(cur), "heading": False, "page": cur_page, "size": body})

    cur_page = first
    for ln in lines:
        rel_top, rel_bot = ln["y0"] / ln["page_h"], ln["y1"] / ln["page_h"]
        raw = plain(ln)
        in_zone = rel_top < a.zone or rel_bot > 1 - a.zone
        if in_zone and (re.sub(r"\d+", "#", raw.lower()) in hf
                        or re.fullmatch(r"[\divxlcm]+", raw, re.I)):
            stats["header_footer_lines"] += 1
            continue
        if is_running_head(ln, raw, offset, a.zone):
            stats["running_heads_by_page_number"] += 1
            continue
        if LICENCE_LINE.search(no_invisible(raw)):
            stats["doi_licence_lines"] += 1
            continue

        size = line_size(ln)
        if size < body * a.small_ratio:
            stats["footnote_lines"] += 1
            continue

        heading = len(raw.split()) <= 12 and (size >= body * 1.15 or (is_bold(ln) and not HEAD_END.search(raw)))
        text = line_text(ln, body, a, stats)
        if not text:
            continue

        if heading:
            flush()
            cur = []
            prev = paras[-1] if paras else None
            # a heading that wraps onto the next line: same page and size, no gap,
            # and the new line does not start a new numbered heading
            if (prev and prev["heading"] and prev["page"] == ln["page"] and prev["size"] == size
                    and -size * 0.5 < ln["y0"] - prev["y1"] < size * 0.6
                    and not re.fullmatch(r"[\dIVXLC]+\.?", prev["text"])
                    and not re.match(r"(\d+(\.\d+)+|box|figure|table|chapter)\b", text, re.I)):
                prev["text"] = join_lines([prev["text"], text])
                prev["y1"] = ln["y1"]
                stats["heading_lines_joined"] += 1
            else:
                paras.append({"text": text, "heading": True, "page": ln["page"], "size": size,
                              "y1": ln["y1"]})
            cur_block = None
            continue

        indent = prev_x0 is not None and ln["block"] == cur_block and ln["x0"] - prev_x0 > ln["page_w"] * 0.015
        new_block = ln["block"] != cur_block
        if cur and (new_block or (indent and SENT_END.search(cur[-1]))):
            flush()
            cur = []
        if not cur:
            cur_page = ln["page"]
        cur.append(text)
        cur_block, prev_x0 = ln["block"], ln["x0"]
    flush()

    # 2. join paragraphs split across pages or blocks
    merged = []
    for p in paras:
        if (merged and not p["heading"] and not merged[-1]["heading"]
                and not SENT_END.search(merged[-1]["text"]) and re.match(r"[a-z(\d]", p["text"])):
            merged[-1]["text"] = join_lines([merged[-1]["text"], p["text"]])
            stats["paragraphs_joined"] += 1
        else:
            merged.append(p)

    # 3. remove back matter, notes sections, captions, tables; clean markers
    out, skipping = [], False
    for p in merged:
        t = p["text"].strip()
        if p["heading"]:
            t = no_invisible(t)
            key = t.strip(" .:").lower()
            if BACK_STOP.match(key) and not a.keep_back_matter:
                stats[f"stopped_at_heading '{t}' (p.{p['page']})"] += 1
                break
            if BACK_SKIP.match(key) and not a.keep_back_matter:
                if not skipping:
                    stats["notes_or_reference_sections_skipped"] += 1
                skipping = True
                continue
            chapter = p["size"] >= body * a.chapter_ratio or CHAPTER_HEAD.match(t)
            if skipping and not chapter:
                stats["notes_paragraphs_removed"] += 1
                continue
            skipping = False
            if a.drop_headings or re.fullmatch(r"(chapter\s+)?[\dIVXLC]+\.?", t, re.I):
                stats["number_only_headings_removed"] += 0 if a.drop_headings else 1
                continue
            out.append(t)
            continue
        if skipping:
            stats["notes_paragraphs_removed"] += 1
            continue
        if CAPTION.match(t) and len(t.split()) < 80:
            stats["captions_removed"] += 1
            continue
        if looks_like_table(t):
            stats["table_like_removed"] += 1
            continue
        t = clean_paragraph(t, a, stats)
        if len(t.split()) < a.min_words:
            stats["very_short_removed"] += 1
            continue
        out.append(t)

    Path(a.out).write_text("\n\n".join(out) + "\n", encoding="utf-8")
    write_report(a.out, [f"source: {a.pdf}  pages {first}-{last}", f"body font size: {body}"],
                 stats, out, counts_only=a.counts_only)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect")
    s.add_argument("pdf")
    s.add_argument("--max-headings", type=int, default=80)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert")
    s.add_argument("pdf")
    s.add_argument("out")
    s.add_argument("--first-page", type=int, help="first page to keep (1-based, as the PDF viewer counts)")
    s.add_argument("--last-page", type=int, help="last page to keep")
    s.add_argument("--zone", type=float, default=0.08, help="top/bottom share of page for headers and footers")
    s.add_argument("--min-repeats", type=int, default=3,
                   help="a line in the header/footer zone that repeats this often (numbers ignored) is removed")
    s.add_argument("--chapter-ratio", type=float, default=1.4,
                   help="a heading at least body size x this is a chapter heading (ends a skipped notes section)")
    s.add_argument("--small-ratio", type=float, default=0.88,
                   help="text smaller than body size x this is footnote text")
    s.add_argument("--min-words", type=int, default=4, help="drop paragraphs shorter than this")
    s.add_argument("--strip-author-year", action="store_true",
                   help="also remove (Smith 2019, p. 4)-style citations; check the report after use")
    s.add_argument("--drop-headings", action="store_true")
    s.add_argument("--keep-back-matter", action="store_true")
    s.add_argument("--counts-only", action="store_true",
                   help="report without text from the book (for sealed books)")
    s.add_argument("--keep-rotated", action="store_true",
                   help="keep rotated (landscape) pages and vertical text lines")
    s.set_defaults(func=cmd_convert)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
