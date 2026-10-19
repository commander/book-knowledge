#!/usr/bin/env python3
"""Check a clean book text file before exam_builder.py split.

  python text_check.py books/book1.txt                # summary + leftovers
  python text_check.py books/book1.txt --sample 20    # also print 20 random paragraphs

It counts words and paragraphs, and searches for text that the converters
must remove (URLs, DOIs, citations, note markers, notes headings, broken
words). Each leftover type shows a count and some examples. A count is not
always an error: read the examples.

epub_to_text.py and pdf_to_text.py use write_report() from this file for
their <out>.report.txt.
"""
import argparse
import random
import re
from pathlib import Path

YEAR = r"(?:1[5-9]\d\d|20\d\d)[a-z]?"
LEFTOVERS = {
    "url": re.compile(r"https?://|www\.\w", re.I),
    "doi": re.compile(r"\bdoi\b|10\.\d{4,}/", re.I),
    "et_al": re.compile(r"\bet al\b", re.I),
    "ibid_op_cit": re.compile(r"\b(?:ibid|op\. cit|loc\. cit)\b", re.I),
    "page_ref": re.compile(r"\bpp?\.\s*\d"),
    "author_year_paren": re.compile(r"\([^()]*?[A-Z][^()]*?\b" + YEAR + r"\b[^()]*\)"),
    "year_page_paren": re.compile(r"\(\s*" + YEAR + r"\s*[,:]\s*(?:pp?\.\s*)?\d+[^()]*\)"),
    "note_marker": re.compile(r"[a-z][.,;:!?)\"'’”](\d{1,3})(?=\s|$)"),
    "bracket_number": re.compile(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]"),
    "notes_heading_line": re.compile(
        r"^(?:notes|endnotes|references|bibliography|works cited|index|further reading)$", re.I),
    "copyright_line": re.compile(r"\bCC[ -]BY\b|made available under|all rights reserved|©", re.I),
    "control_char": re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]"),
    "invisible_char": re.compile("[\u00ad\u200b\u200c\u200d\ufeff]"),
    "hyphen_space": re.compile(r"\b[a-z]{2,}- [a-z]{2,}\b"),
}
SHOW = 5

# One author-year item: "Marino 2020", "van Dijck et al. 2018", "Kunz, 1973",
# "Hiemstra 2019: 55", "Human Rights Watch 2016a".
_NAME = r"(?:[a-z]{1,3}\s+)?[A-ZÀ-Þ][\w'\-.]*"
_JOIN = r"(?:and|&|et al\.?|de|del|la|le|van|von|der|da|di|y)"
AUTHOR_YEAR = (rf"\b{_NAME}(?:\s+(?:{_NAME}|{_JOIN}))*,?\s+{YEAR}\b"
               r"(?:\s*[,:]\s*(?:pp?\.\s*)?\d+(?:\s*[\-–]\s*\d+)?)?")
_FILLER = (r"\b(?:see|also|cf|e\.g|i\.e|and|for example|for instance|pp?|in|as|by|"
           r"described|discussed|quoted|cited)\b\.?")
_MONTH = (r"\b(?:January|February|March|April|May|June|July|August|September|"
          r"October|November|December)\s+" + YEAR)


def drop_citation_groups(t, stats):
    """Remove parentheses that hold only author-year citations.

    "(Marino 2020)", "(e.g. Tufte 2016, Barr 2012)", "Burke (1978, 2014)" are
    removed. "(Personal interview, May 2016)" and "(in the mid-2000s)" stay,
    because other words are left after the citations are removed.
    """
    def paren(m):
        inner = m.group(1)
        if not re.search(YEAR, inner) or re.search(_MONTH, inner):
            return m.group(0)                       # no year, or a date such as "23 July 2017"
        rest = re.sub(AUTHOR_YEAR, " ", inner)
        if rest == inner and not re.match(r"\s*" + YEAR + r"(?:[\s,;:\-–]+(?:\d+|" + YEAR + r"))*\s*$", inner):
            return m.group(0)
        if rest == inner and not re.search(r"(?:[A-Z][\w'\-]*|al\.)\s*$", t[:m.start()]):
            return m.group(0)                       # "(2004)" after a title, not after a name
        rest = re.sub(_FILLER, " ", rest, flags=re.I)
        if re.sub(r"[\d\W_]+", " ", rest).strip():
            return m.group(0)
        stats["author_year_citations_removed"] += 1
        return ""

    t = re.sub(r"\s?\(([^()]*)\)", paren, t)
    return re.sub(r"\s+([.,;:!?)])", r"\1", t)


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def find_leftovers(paras, show=SHOW):
    """Return {name: (count, [examples])} for the leftover types found."""
    found = {}
    for name, rx in LEFTOVERS.items():
        count, examples = 0, []
        for p in paras:
            for m in rx.finditer(p):
                count += 1
                if len(examples) < show:
                    a, b = max(0, m.start() - 50), min(len(p), m.end() + 50)
                    examples.append(("..." if a else "") + p[a:b].replace("\n", " ")
                                    + ("..." if b < len(p) else ""))
        if count:
            found[name] = (count, examples)
    return found


def summary_lines(paras, counts_only=False):
    words = [len(p.split()) for p in paras]
    short = sum(w < 8 for w in words)
    lines = [f"paragraphs: {len(paras)}   words: {sum(words)}   "
             f"paragraphs under 8 words (mostly headings): {short}"]
    found = find_leftovers(paras)
    if not found:
        lines.append("leftovers: none found")
        return lines
    lines.append("leftovers (count)" if counts_only else "leftovers (count, then examples):")
    for name, (count, examples) in found.items():
        lines.append(f"  {name}: {count}")
        if not counts_only:
            lines += [f"      | {e}" for e in examples]
    return lines


def sample_lines(paras, n, seed=0):
    body = [p for p in paras if len(p.split()) > 30]
    rng = random.Random(seed)
    out = [f"{n} random paragraphs (check them):", ""]
    for p in rng.sample(body, min(n, len(body))):
        out += [p, "-" * 60]
    return out


def write_report(out_path, header, stats, paras, n_sample=20, counts_only=False):
    """Write <out>.report.txt and print the top part. Return the report path.

    counts_only: no text from the book in the report (for sealed books).
    """
    lines = list(header) + summary_lines(paras, counts_only) + ["", "removed / changed:"]
    lines += [f"  {k}: {v}" for k, v in stats.most_common()
              if not (counts_only and k.startswith("stopped_at_heading"))] or ["  (nothing)"]
    top = len(lines)
    if not counts_only:
        lines += [""] + sample_lines(paras, n_sample)
    rpath = Path(out_path).with_suffix(".report.txt")
    rpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:top]))
    print(f"\nreport: {rpath}")
    return rpath


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("txt")
    p.add_argument("--sample", type=int, default=0, help="print this many random paragraphs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    paras = paragraphs(Path(a.txt).read_text(encoding="utf-8"))
    print("\n".join(summary_lines(paras)))
    if a.sample:
        print()
        print("\n".join(sample_lines(paras, a.sample, a.seed)))


if __name__ == "__main__":
    main()
