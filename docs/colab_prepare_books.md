# Colab guide: books to clean text and units

This guide prepares the books for the exam builder. For each book, the
pipeline converts the EPUB or PDF to clean text and splits the text into
units. One config file (`books.json`) controls all books. To change the
books, edit the config and run the same command again.

This step does not need a GPU. Use a **CPU runtime** to save compute units.

## 1. Put the project on Google Drive (one time)

Make this folder on Google Drive. Use the same names.

```
MyDrive/book-knowledge/
  books.json               config: which books, which options
  requirements-prep.txt    pinned package versions
  src/                     epub_to_text.py, pdf_to_text.py, text_check.py,
                           prepare_books.py, exam_builder.py
  prompts/                 knowledge_prompt.txt, trap_twin_prompt.txt
  data/                    everyday.jsonl
  books/                   the source files (.epub, .pdf)
```

- Copy these items from the repo on your Mac. Do not copy `src/.venv` or `src/__pycache__`.
- Put the book files in `books/`. You do not need the old `books/*.txt` and
  `books/*.report.txt` files. The pipeline writes its output in `work/`.
- When you change a script on the Mac, copy it to Drive again.

## 2. Start the notebook

Open a new Colab notebook. Select **Runtime > Change runtime type > CPU**.

Cell 1: connect Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Cell 2: go to the project folder and install the packages. Do this cell
again each time the runtime starts.

```python
%cd /content/drive/MyDrive/book-knowledge
!pip install -q -r requirements-prep.txt
!ls books
```

## 3. Examine each new book (only when you add a book)

Do this step for an open book only. **Do not do it for a sealed book** (see
section 8).

**EPUB.** List the files in reading order. The `keep` column shows which files
the converter uses. Make sure that `yes` is set for the introduction and all
chapters, and `-` is set for the front matter, bibliography and index.

```python
!python src/epub_to_text.py list books/organized-crime-and-migration-victorie-knox.epub
```

If a body file has `-`, add `"keep": ["<file name regex>"]` to the options of the book.
If a front matter file has `yes`, add `"skip": ["<file name regex>"]`.

**PDF.** Find the body font size, and find the pages where the main text
starts and stops.

```python
!python src/pdf_to_text.py inspect books/critical-software-studies.pdf --max-headings 200
```

- `first_page`: the page of the first chapter heading (or the introduction, if
  you want it). Do not include the contents, the preface or the series pages.
- `last_page`: the last page before the index. If the book has no index, use
  the last page. Notes and reference lists after each chapter are skipped
  automatically. You do not need to remove them with the page range.

Use the EPUB when a book has both an EPUB and a PDF. The EPUB gives cleaner
text.

## 4. Write the config

Open `books.json` in the Colab file browser (double-click to edit it) and save
it. Or write it from a cell:

```python
%%writefile books.json
{
  "books_dir": "books",
  "work_dir": "work",
  "split": {"min_words": 150, "max_words": 250},
  "books": [
    {
      "name": "critical-software-studies",
      "title": "Critical Software Studies",
      "source": "critical-software-studies.pdf",
      "options": {"first_page": 18, "last_page": 247, "strip_author_year": true}
    },
    {
      "name": "organized-crime-and-migration",
      "title": "Organized Crime and Migration",
      "source": "organized-crime-and-migration-victorie-knox.epub",
      "options": {}
    },
    {
      "name": "literature-for-change",
      "title": "Literature for Change",
      "source": "literature-for-change-charles-burdett.epub",
      "options": {}
    }
  ]
}
```

Rules:

- `name`: a short, unique name. The output goes to `work/<name>/`.
- `source`: the file name in `books/`.
- `title`: the title for the exam prompts later.
- `options`: the converter options. Write each option without `--`, and with
  `_` in place of `-`. For example, `--first-page 18` is `"first_page": 18`, and
  `--strip-author-year` is `"strip_author_year": true`.

Options that you will use most:

| Option | EPUB | PDF | What it does |
|---|---|---|---|
| `first_page`, `last_page` | | yes | Page range of the main text |
| `strip_author_year` | yes | yes | Removes citations such as "(Smith 2019, p. 4)". Use it for academic books. |
| `keep`, `skip` | yes | | Lists of file name patterns to include or exclude |
| `keep_citations` | yes | | Keeps citations that link to the bibliography (they are removed by default) |
| `keep_epigraphs` | yes | | Keeps chapter epigraphs |
| `drop_headings` | yes | yes | Removes headings from the text |
| `chapter_ratio` | | yes | Heading size (x body size) that starts a new chapter. Default 1.4. |
| `min_repeats` | | yes | Repeat count for a running header or footer. Default 3. |
| `keep_rotated` | | yes | Keeps rotated (landscape) pages. They are skipped by default (see below). |
| `small_ratio` | | yes | Text smaller than this x body size is footnote text. Default 0.88. |

## 5. Run the pipeline

```python
!python src/prepare_books.py --config books.json
```

It prints one row for each book:

```
book                                          status    words  units  leftovers
critical-software-studies                     ok        89477    366  et_al 5, author_year_paren 1, hyphen_space 2
organized-crime-and-migration                 ok        75764    335  et_al 5, author_year_paren 6, hyphen_space 1
literature-for-change                         ok        59166    256  author_year_paren 10, note_marker 1, hyphen_space 2
```

These are the expected numbers for the three current books. If you get the
same numbers, the run is correct.

To run one book only (for example, after you change its options):

```python
!python src/prepare_books.py --config books.json --only literature-for-change
```

Output for each book, in `work/<name>/`:

| File | Content |
|---|---|
| `book.txt` | Clean text: one paragraph per block, headings on their own line |
| `book.report.txt` | Removal counts, leftovers with examples, 20 random paragraphs |
| `units.jsonl` | The units (`U0001`, ...) for `exam_builder.py` |
| `convert.log` | Full output of the converter and the split |
| `prepare.json` | Record of the run: source file hash, options, script hashes, package versions, counts |

`work/prepare_summary.tsv` has the table for all books.

**Known limitation: tables.** `pdf_to_text.py` skips rotated (landscape)
pages, for example a wide table printed sideways. The report lists them as
`rotated_page_skipped p.<n>`. A table on a normal page can have its columns
mixed. `epub_to_text.py` removes EPUB tables. Proper table parsing is planned
for the next iteration (`agent.md` section 5c).

## 6. Check each book

Open `work/<name>/book.report.txt`. Do these checks:

1. **Word count.** It must be near the body length of the book. Notes and
   reference lists are removed, so the count is lower than the full book.
   A very low count shows that a part of the book is missing.
2. **Leftovers.** Read the examples. Some leftovers are correct text. For
   example, "Zetter et al. (2013) note that ..." is a sentence, not a
   citation, and "post- and de-colonial" is correct English. Many URLs, DOIs,
   "References" lines or note numbers show a problem.
3. **20 random paragraphs.** Each paragraph must be clean body text: no page
   numbers, no running heads, no notes, no words split in two.
4. **Units.** Book 1 needs two exams (dev and acceptance) from two separate
   unit pools. Use a book with about 300 units or more as book 1. A book with
   one exam needs about 200 units or more.

To see more paragraphs:

```python
!python src/text_check.py work/literature-for-change/book.txt --sample 10 --seed 1
```

If you find a problem, change the options in `books.json` and run section 5
again. If an option cannot fix it, tell Claude, and include the problem text and the
book name.

## 7. Repeat with other books

1. Put the new file in `books/`.
2. Do section 3 for the new book.
3. Add the book to `books.json`, or replace a book in it.
4. Run section 5. Run section 6.

The same config and the same source files always give the same `units.jsonl`.
`prepare.json` records the hashes, so you can show which files and options
made the units.

## 8. Sealed books

For a sealed book, you must not see the text.

- Add `"sealed": true` to the book in `books.json`. The report then has counts only
  (no examples, no sample paragraphs). The summary table prints counts only.
- Do not run `inspect` or `list` on it. Do not open `book.txt`, `units.jsonl` or `convert.log`.
- Use an EPUB if possible. A PDF needs a page range, and you can only find
  the range when you look at the book. A PDF without a page range keeps the
  front matter (contents, preface).
- The design (section 2.4) says that a script must select the sealed books.
  That script can write the sealed entries into the config.

## 9. Next step

When all books pass section 6, do the familiarity check. The full guide for
all phases is `docs/colab_full_pipeline.md`.
