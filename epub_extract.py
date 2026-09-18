#!/usr/bin/env python3
"""Extract a Vellum-built EPUB into ordered chapters of paragraphs (stdlib only).

Output: list of {"title": str, "paras": [str, ...]} — italics kept as *...* markers,
scene breaks as the literal line "***". Front/back matter (copyright, also-by, TOC,
about the author, newsletter pages) is dropped by heuristic.
"""
import re, sys, json, zipfile, html, posixpath
from html.parser import HTMLParser

FRONT_BACK = re.compile(r"(copyright|all rights reserved|also by|about the author|table of contents|"
                        r"newsletter|sign up|acknowledg|dedication|other books|preview|excerpt|"
                        r"contents|isbn|published by|love in a book)", re.I)

class _P(HTMLParser):
    BLOCK = {"p", "div", "h1", "h2", "h3", "h4", "li", "blockquote", "hr", "br", "section"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []      # (kind, text)
        self.cur = []
        self.kind = "p"
        self.ital = 0
        self.skip = 0
    def _flush(self):
        t = "".join(self.cur).strip()
        t = re.sub(r"[ \t\r\n]+", " ", t)
        if t:
            self.blocks.append((self.kind, t))
        self.cur = []; self.kind = "p"
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "script", "head", "title"):
            self.skip += 1; return
        if tag in self.BLOCK:
            self._flush()
            if tag in ("h1", "h2", "h3", "h4"):
                self.kind = "h"
            if tag == "hr":
                self.blocks.append(("break", "***"))
            cls = (a.get("class") or "") + " " + (a.get("epub:type") or "")
            if re.search(r"(scene-break|ornament|separator|implicit-break)", cls, re.I) and tag in ("p", "div"):
                self.blocks.append(("break", "***"))
        if tag in ("i", "em"):
            self.ital += 1; self.cur.append("*")
    def handle_endtag(self, tag):
        if tag in ("style", "script", "head", "title"):
            self.skip = max(0, self.skip - 1); return
        if tag in ("i", "em") and self.ital:
            self.ital -= 1; self.cur.append("*")
        if tag in self.BLOCK:
            self._flush()
    def handle_data(self, data):
        if self.skip: return
        self.cur.append(data)

def _spine(z):
    cont = z.read("META-INF/container.xml").decode("utf-8", "ignore")
    opf = re.search(r'full-path="([^"]+)"', cont).group(1)
    base = posixpath.dirname(opf)
    o = z.read(opf).decode("utf-8", "ignore")
    items = dict(re.findall(r'<item[^>]+id="([^"]+)"[^>]+href="([^"]+)"', o))
    items.update({k: v for v, k in re.findall(r'<item[^>]+href="([^"]+)"[^>]+id="([^"]+)"', o)})
    order = re.findall(r'<itemref[^>]+idref="([^"]+)"', o)
    out = []
    for idref in order:
        href = items.get(idref)
        if href and re.search(r"\.x?html?$", href, re.I):
            out.append(posixpath.normpath(posixpath.join(base, href)) if base else href)
    return out

def extract(path):
    z = zipfile.ZipFile(path)
    chapters = []
    spine = _spine(z)
    chapter_files = [f for f in spine if re.search(r"(chapter-?\d+|epilogue|prologue)", posixpath.basename(f), re.I)]
    if chapter_files:
        spine = chapter_files  # Vellum names chapter files; everything else is front/back matter
    for f in spine:
        try:
            raw = z.read(f).decode("utf-8", "ignore")
        except KeyError:
            continue
        p = _P(); p.feed(raw); p._flush()
        blocks = p.blocks
        if not blocks:
            continue
        heads = [t for k, t in blocks if k == "h"]
        body = [(k, t) for k, t in blocks if k != "h"]
        words = sum(len(t.split()) for k, t in body)
        text_head = " ".join(heads[:3])
        if FRONT_BACK.search(text_head):
            continue
        if words < 250 and (FRONT_BACK.search(text_head) or FRONT_BACK.search(" ".join(t for k, t in body)[:600])):
            continue
        if words < 60:
            continue
        # Vellum chapter heading: h1 "Chapter One" + optional subtitle; join them
        title = " — ".join(h for h in heads[:2]) if heads else f"Chapter {len(chapters)+1}"
        paras = []
        for k, t in body:
            if k == "break":
                if paras and paras[-1] != "***":
                    paras.append("***")
            else:
                # normalise italics markers that got split oddly
                t = re.sub(r"\*\s+\*", "", t)
                t = re.sub(r"\s+", " ", t).strip()
                if t:
                    paras.append(t)
        while paras and paras[-1] == "***":
            paras.pop()
        chapters.append({"title": title, "paras": paras, "words": words, "src": f})
    return chapters

if __name__ == "__main__":
    ch = extract(sys.argv[1])
    print(json.dumps({"chapters": len(ch), "words": sum(c["words"] for c in ch),
                      "titles": [c["title"] for c in ch]}, ensure_ascii=False, indent=1))
