"""Rebuild train.txt from your own books (optional).

Put .txt, .pdf or .html files into a folder called "books" next to this
script, then run:   python prepare.py

Your current train.txt is already finished. This script only matters if you
want to build a new one. It overwrites train.txt. The old text is first
copied to train_old.txt. If the characters in the new text differ, train.py
asks you to use --fresh.
"""
import re, shutil, sys
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).resolve().parent
BOOKS, OUT = BASE / "books", BASE / "train.txt"


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def read_file(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader          # pip install pypdf
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if ext in (".html", ".htm"):
        parser = TextOnly()
        parser.feed(raw)
        return "".join(parser.parts)
    return raw


def strip_gutenberg(t):
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t, re.I | re.S)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t, re.I)
    if m:
        t = t[:m.start()]
    return t


def clean(t):
    t = strip_gutenberg(t)
    t = re.sub(r"(\w)[-\u2010\u2011\u2212\u2013]\n(\w)", r"\1\2", t)   # "possi-\nble"
    t = re.sub(r"(fi|fl) (?=[a-z])", r"\1", t)                          # "fi rst"
    t = re.sub(r"(?m)^\s*\d+\s*$", "", t)                               # page numbers
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", t)]
    return "\n\n".join(p for p in paras if p)


if not BOOKS.is_dir():
    sys.exit(f'No "books" folder at {BOOKS}.\nCreate it and put your books inside first.')

parts = []
for path in sorted(BOOKS.iterdir()):
    if not path.is_file():
        continue
    try:
        text = clean(read_file(path))
    except Exception as e:
        print("SKIPPED", path.name, "-", e)
        continue
    print(f"{path.name}: {len(text):,} characters")
    if len(text) > 1000:
        parts.append(text)

if not parts:
    sys.exit("No usable text found, train.txt was not changed.")
if OUT.exists():
    shutil.copy(OUT, BASE / "train_old.txt")
full = "\n\n".join(parts)
OUT.write_text(full, encoding="utf-8")
print(f"TOTAL: {len(full):,} characters from {len(parts)} files -> train.txt")
