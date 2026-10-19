#!/usr/bin/env python3
"""Convert a book EPUB to clean plain text for exam_builder.py.

Step 1 - list the spine files (reading order), with word counts and the
default keep/skip decision:
  python epub_to_text.py list book.epub

Step 2 - convert the kept files:
  python epub_to_text.py convert book.epub book.txt

Step 3 - read book.report.txt (removed items, leftovers, 20 random paragraphs).

Spine files are skipped by file name (cover, title, copyright, contents, nav,
series, index, bibliography, contributors, acknowledgments, ...). Use --skip
and --keep (regular expressions on the file name) to change this.

What "convert" removes from the kept files:
  - note markers (epub:type/role "noteref", and <sup> with a link or a number)
  - footnote and endnote sections, bibliography and reference sections
  - page-break markers, figures, captions, images, tables
  - epigraphs (use --keep-epigraphs to keep them)
  - author bylines (a paragraph that is only a link to a skipped file,
    for example a link to contributors.xhtml)
  - author-year citations that link to the bibliography, for example
    "(Smith 2010; Jones 2012: 44)". If the parentheses contain more text,
    only the linked citations are removed. Use --keep-citations to keep them.
  - with --strip-author-year: also citations without a link, when the
    parentheses hold only citations
What it writes:
  - one paragraph per block, separated by a blank line
  - headings on their own line (use --drop-headings to remove them)

Requires: pip install beautifulsoup4 lxml
"""
import argparse
import posixpath
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString

from text_check import drop_citation_groups, write_report

SKIP_FILES = re.compile(
    r"cover|title|copyright|contents|\btoc\b|^toc|nav|series|index|bibliograph|"
    r"references|contributor|acknowledg|dedication|halftitle|half-title|"
    r"about|also-?by|abbreviation|glossary|colophon|praise", re.I)

# epub:type / role tokens of elements that are removed with all their content
DROP_TYPES = {
    "noteref", "doc-noteref", "footnote", "footnotes", "doc-footnote",
    "endnote", "endnotes", "doc-endnote", "doc-endnotes", "rearnote", "rearnotes",
    "bibliography", "doc-bibliography", "biblioentry", "doc-biblioentry",
    "pagebreak", "doc-pagebreak", "index", "doc-index", "toc", "doc-toc",
    "backlink", "doc-backlink",
}
EPIGRAPH_TYPES = {"epigraph", "doc-epigraph"}
DROP_TAGS = ["script", "style", "figure", "figcaption", "img", "svg", "table", "math"]
BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd",
              "blockquote", "div", "pre", "section", "aside"]
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CITE = "\u2063"     # invisible marker around linked citations while cleaning
# "Slack and Whiteford 2013", "Human Rights Watch 2016a"; used only inside a
# parenthesis group that also has linked citations
_NAME = r"[A-ZÀ-Þ][\w'\-.]*"
_JOIN = r"(?:and|&|et al\.?|de|del|la|le|van|von|der|da|di|y)"
UNLINKED_CITATION = rf"\b{_NAME}(?:\s+(?:{_NAME}|{_JOIN}))*\s+(?:1[5-9]\d\d|20\d\d)[a-z]?\b"


# ---------------------------------------------------------------- reading

def read_spine(z):
    """Return [(idref, path in zip)] in reading order."""
    container = BeautifulSoup(z.read("META-INF/container.xml"), "lxml-xml")
    opf_path = container.find("rootfile")["full-path"]
    opf = BeautifulSoup(z.read(opf_path), "lxml-xml")
    base = posixpath.dirname(opf_path)
    manifest = {it["id"]: posixpath.normpath(posixpath.join(base, unquote(it["href"])))
                for it in opf.find("manifest").find_all("item")}
    return [(ref["idref"], manifest[ref["idref"]])
            for ref in opf.find("spine").find_all("itemref") if ref["idref"] in manifest]


def load(z, path):
    return BeautifulSoup(z.read(path), "lxml-xml")


def tokens(el):
    """epub:type and role values of an element, as a set of words."""
    out = set()
    for key in ("epub:type", "type", "role"):
        v = el.get(key)
        if v:
            out.update(v.lower().split())
    return out


def keep_file(path, a):
    name = posixpath.basename(path)
    if any(re.search(k, name, re.I) for k in a.keep):
        return True
    if any(re.search(s, name, re.I) for s in a.skip):
        return False
    return not SKIP_FILES.search(posixpath.splitext(name)[0])


def norm_text(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub("[\u00ad\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", s.replace("\t", " "))   # control characters
    for x, y in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(x, y)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- cleaning

def link_target(a_tag, doc_path):
    href = a_tag.get("href") or ""
    if not href or re.match(r"[a-z]+:", href):
        return None, None
    file_part, _, frag = href.partition("#")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(doc_path), unquote(file_part))) \
        if file_part else doc_path
    return target, frag


def collect_bib_ids(z, spine):
    """Ids of elements inside bibliography sections, per file."""
    ids = {}
    for _, path in spine:
        soup = load(z, path)
        for sec in soup.find_all(lambda t: tokens(t) & {"bibliography", "doc-bibliography"}):
            ids.setdefault(path, set()).update(
                el["id"] for el in sec.find_all(attrs={"id": True}))
            if sec.get("id"):
                ids[path].add(sec["id"])
    return ids


def clean_doc(soup, path, kept_paths, bib_ids, a, stats):
    body = soup.find("body") or soup

    # 1. bylines: a paragraph that is only a link to a skipped file
    for p in body.find_all("p"):
        links = p.find_all("a", href=True)
        text = norm_text(p.get_text())
        if links and len(text.split()) <= 8 and norm_text("".join(x.get_text() for x in links)) == text:
            target, _ = link_target(links[0], path)
            if target and target not in kept_paths and not BIB_FILE.search(target):
                p.decompose()
                stats["bylines_removed"] += 1

    # 2. linked citations -> marked text (removed later, in clean_paragraph)
    if not a.keep_citations:
        for x in body.find_all("a", href=True):
            target, frag = link_target(x, path)
            if target and (BIB_FILE.search(posixpath.basename(target))
                           or frag in bib_ids.get(target, ())):
                x.replace_with(CITE + x.get_text() + CITE)
                stats["linked_citations_marked"] += 1

    # 3. elements removed with their content
    drop = set(DROP_TYPES) | (set() if a.keep_epigraphs else EPIGRAPH_TYPES)
    for el in body.find_all(lambda t: bool(tokens(t) & drop)):
        if el.decomposed:
            continue
        kind = sorted(tokens(el) & drop)[0].replace("doc-", "")
        stats[f"removed_{kind}"] += 1
        el.decompose()
    for el in body.find_all(DROP_TAGS):
        if not el.decomposed:
            stats[f"removed_{el.name}"] += 1
            el.decompose()
    for el in body.find_all("sup"):
        t = el.get_text().strip()
        if el.find("a") or re.fullmatch(r"[\d*†‡,\-– ]+", t):
            stats["removed_sup_note_marker"] += 1
            el.decompose()
    for cls in a.drop_class:
        for el in body.find_all(class_=cls):
            stats[f"removed_class_{cls}"] += 1
            el.decompose()

    # 4. line breaks inside a block become spaces
    for br in body.find_all("br"):
        br.replace_with(" ")
    return body


BIB_FILE = re.compile(r"bibliograph|references", re.I)


def leaf_blocks(body):
    """Block elements with text that contain no other block element."""
    out = []
    for el in body.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):
            # text directly in a container next to child blocks: keep it too
            loose = "".join(c for c in el.children if isinstance(c, NavigableString)).strip()
            if loose and len(loose.split()) > 3:
                out.append((el.name, loose))
            continue
        out.append((el.name, el.get_text()))
    return out


def strip_citations(t, stats):
    """Remove linked citations, marked as CITE...CITE."""
    if CITE not in t:
        return t

    def paren(m):
        inner = m.group(1)
        # remove linked citations, then unlinked "Name 2015" items in the same group
        inner = re.sub(CITE + r"[^" + CITE + r"]*" + CITE + r"[\s,;:\d\-–]*", " ", inner)
        inner = re.sub(UNLINKED_CITATION + r"[\s,;:\d\-–]*", " ", inner).replace(CITE, "")
        rest = re.sub(r"\b(?:see|also|cf|e\.g|i\.e|and|as|in|by|described|discussed|"
                      r"pp?|et al|de|van|von|der|da|del|di|le|la)\b\.?", " ", inner, flags=re.I)
        rest = re.sub(r"[\d\W_]+", " ", rest).strip()
        pointer = re.match(r"[\s,;:]*(?:see\b|for (?:a |an )?(?:more |further |detailed |full )?"
                           r"(?:analysis|discussion|details?|overview|review|account))", inner, re.I)
        if len(rest.split()) <= 2 or pointer:
            stats["citation_groups_removed"] += 1
            return ""
        stats["citations_removed_inside_text"] += 1
        # "..., see, for instance," is left when the citations were at the end
        inner = re.sub(r"(?:[\s,;:]*\b(?:see|also|cf\.?|e\.g\.?|for instance|for example))+[\s,;:.]*$",
                       "", inner, flags=re.I)
        return "(" + inner.rstrip(" ,;:") + ")"

    t = re.sub(r"\(([^()]*" + CITE + r"[^()]*)\)", paren, t)
    # a citation outside parentheses is part of the sentence ("Smith 2010 argues"): keep the text
    return t.replace(CITE, "")


def clean_paragraph(t, stats):
    before = t
    t = re.sub(r"^[•●▪–\-*]\s*", "", t)          # list bullets
    t = re.sub(r"\(\s*[,;:]*\s*\)", "", t)
    t = re.sub(r"\s+([.,;:!?)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if t != before:
        stats["paragraphs_tidied"] += 1
    return t


# ---------------------------------------------------------------- commands

def cmd_list(a):
    z = zipfile.ZipFile(a.epub)
    spine = read_spine(z)
    print(f"{'keep':<5} {'words':>7}  file")
    total = 0
    for _, path in spine:
        k = keep_file(path, a)
        soup = load(z, path)
        n = len(norm_text((soup.find("body") or soup).get_text(" ")).split())
        total += n if k else 0
        print(f"{'yes' if k else '-':<5} {n:>7}  {path}")
    print(f"\nwords in kept files (before cleaning): {total}")


def cmd_convert(a):
    z = zipfile.ZipFile(a.epub)
    spine = read_spine(z)
    kept = [p for _, p in spine if keep_file(p, a)]
    if not kept:
        raise SystemExit("No spine file is kept. Run 'list' and use --keep.")
    kept_paths = set(kept)
    bib_ids = collect_bib_ids(z, spine)
    stats = Counter()
    out = []
    for path in kept:
        body = clean_doc(load(z, path), path, kept_paths, bib_ids, a, stats)
        n_before = len(out)
        for tag, raw in leaf_blocks(body):
            t = strip_citations(norm_text(raw), stats)
            if a.strip_author_year:
                t = drop_citation_groups(t, stats)
            t = clean_paragraph(t, stats)
            if not t:
                continue
            if tag in HEADINGS:
                if a.drop_headings:
                    stats["headings_removed"] += 1
                    continue
                out.append(t)
                continue
            if len(t.split()) < a.min_words:
                stats["very_short_removed"] += 1
                continue
            out.append(t)
        stats[f"file {posixpath.basename(path)}: paragraphs"] = len(out) - n_before

    Path(a.out).write_text("\n\n".join(out) + "\n", encoding="utf-8")
    skipped = [posixpath.basename(p) for _, p in spine if p not in kept_paths]
    header = [f"source: {a.epub}",
              f"files kept: {', '.join(posixpath.basename(p) for p in kept)}",
              f"files skipped: {', '.join(skipped)}"]
    write_report(a.out, header, stats, out, counts_only=a.counts_only)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def file_options(s):
        s.add_argument("--keep", nargs="*", default=[], help="regex: always keep these spine files")
        s.add_argument("--skip", nargs="*", default=[], help="regex: also skip these spine files")

    s = sub.add_parser("list")
    s.add_argument("epub")
    file_options(s)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("convert")
    s.add_argument("epub")
    s.add_argument("out")
    file_options(s)
    s.add_argument("--min-words", type=int, default=4, help="drop paragraphs shorter than this")
    s.add_argument("--drop-headings", action="store_true")
    s.add_argument("--keep-epigraphs", action="store_true")
    s.add_argument("--keep-citations", action="store_true",
                   help="keep author-year citations that link to the bibliography")
    s.add_argument("--strip-author-year", action="store_true",
                   help="also remove citations without a link, for example (Smith 2019, p. 4); "
                        "check the report after use")
    s.add_argument("--counts-only", action="store_true",
                   help="report without text from the book (for sealed books)")
    s.add_argument("--drop-class", nargs="*", default=[],
                   help="also remove elements with these CSS classes")
    s.set_defaults(func=cmd_convert)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
