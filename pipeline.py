#!/usr/bin/env python3
"""Novel translation pipeline: English to German / Brazilian Portuguese / Spanish / Italian / French on free NVIDIA NIM models.

Per book × language:  extract → chunk → translate (model ladder, keep trying) → lint (Test A)
                      → two judges from other model families (Test B) → line fixes → re-lint
                      → Vellum-ready .docx.   Checkpointed per chunk; resumable; quiet.

Usage:
  python3 pipeline.py run --series steel --langs de,pt-BR,es,it,fr --books all
  python3 pipeline.py run --series steel --langs de --books "Rocking Player"
  python3 pipeline.py status --series steel
Output tree: books/<series>/<book>/<lang>/{chunks.jsonl, lint.json, judge.jsonl, fixes.jsonl, summary.json, <Book>-<Lang>.docx}
"""
import os, re, sys, json, time, glob, argparse, random, threading, urllib.request, urllib.error
import concurrent.futures as cf
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.environ.get("SOURCE_DIR", os.path.join(ROOT, "source"))  # one folder per book, containing that book's EPUB
sys.path.insert(0, ROOT)
from epub_extract import extract

# ---------------------------------------------------------------- config
LANG_NAME = {"de": "German", "pt-BR": "Brazilian Portuguese", "es": "Spanish (neutral Latin American)",
             "it": "Italian", "fr": "French"}
LANG_LABEL = {"de": "German", "pt-BR": "Portuguese", "es": "Spanish", "it": "Italian", "fr": "French"}
CHAPTER_WORD = {"de": ("Kapitel", "Epilog", "Prolog"), "pt-BR": ("Capítulo", "Epílogo", "Prólogo"),
                "es": ("Capítulo", "Epílogo", "Prólogo"), "it": ("Capitolo", "Epilogo", "Prologo"),
                "fr": ("Chapitre", "Épilogue", "Prologue")}
# formal / informal "you" forms per language for the register count (Test A)
FORMAL = {"de": r"\b(Sie|Ihnen|Ihr|Ihre|Ihrer|Ihrem|Ihren)\b", "pt-BR": r"\b(o senhor|a senhora|senhor|senhora)\b",
          "es": r"\b(usted|ustedes)\b", "it": r"\b(Lei|Le|La)\b", "fr": r"\b(vous|votre|vos)\b"}
INFORMAL = {"de": r"\b(du|dich|dir|dein|deine|deinen|deinem|deiner)\b", "pt-BR": r"\b(você|te|teu|tua|seu|sua)\b",
            "es": r"\b(tú|te|ti|tu|tus|contigo)\b", "it": r"\b(tu|te|ti|tuo|tua|tuoi|tue)\b", "fr": r"\b(tu|te|toi|ton|ta|tes)\b"}
QUOTE_OK = {"de": lambda t: "„" in t, "pt-BR": lambda t: "—" in t or "“" in t, "es": lambda t: "—" in t or "«" in t or "“" in t,
            "it": lambda t: "«" in t or "“" in t, "fr": lambda t: "«" in t or "—" in t}
BANNED = {"de": [r"\bMs\.", r"\bMr\.", r"\bMrs\.", r"einen Fuß (von|entfernt)", r"\bFräulein\b"],
          "pt-BR": [r"\bMs\.", r"\bMr\.", r"\bMrs\.", r"\bestou a \w+ar\b", r"\brapariga\b"],
          "es": [r"\bMs\.", r"\bMr\.", r"\bMrs\.", r"\bvosotr[oa]s\b", r"\bos\b\s+(quiero|amo|dije)"],
          "it": [r"\bMs\.", r"\bMr\.", r"\bMrs\."],
          "fr": [r"\bMs\.", r"\bMr\.", r"\bMrs\."]}

TRANSLATORS = ["google/gemma-4-31b-it", "moonshotai/kimi-k3", "deepseek-ai/deepseek-v4-flash-0731",
               "z-ai/glm-5.3-flash"]
JUDGES_POOL = ["moonshotai/kimi-k3", "deepseek-ai/deepseek-v4-flash-0731", "google/gemma-4-31b-it",
               "nvidia/nemotron-3.5-lightning-30b-a3b"]
CHUNK_WORDS = 1000
WORKERS = 6
API = "https://integrate.api.nvidia.com/v1/chat/completions"

def _keys():
    ks = []
    for l in open(os.path.join(ROOT, ".env")):
        if l.startswith("NVIDIA_API_KEY_") and "=" in l:
            v = l.split("=", 1)[1].strip()
            if v: ks.append(v)
    if not ks: raise SystemExit("no NVIDIA_API_KEY_* in .env: copy .env.example to .env and add your own free keys from build.nvidia.com")
    return ks
KEYS = _keys()
_key_i = [0]; _lock = threading.Lock()
def _key():
    with _lock:
        _key_i[0] = (_key_i[0] + 1) % len(KEYS); return KEYS[_key_i[0]]

LOG = open(os.path.join(ROOT, "logs", f"run-{datetime.now():%Y%m%d-%H%M%S}.log"), "a")
def log(*a):
    s = f"{datetime.now():%H:%M:%S} " + " ".join(str(x) for x in a)
    LOG.write(s + "\n"); LOG.flush()

# ---------------------------------------------------------------- llm call with ladder
def _stream(model, system, user, max_tokens, temperature, timeout):
    """Streaming call: returns (text, finish_reason). Streaming keeps NVIDIA's gateway from 504-ing on long outputs."""
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # reasoning models (deepseek, glm, nemotron) otherwise spend the whole budget thinking and return nothing
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}, "reasoning_effort": "none"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    parts, fin = [], None
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        try: j = json.loads(data)
        except Exception: continue
        for ch in j.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("content"): parts.append(d["content"])
            if ch.get("finish_reason"): fin = ch["finish_reason"]
    return "".join(parts), fin

def chat(models, system, user, max_tokens=8192, temperature=0.3, tries_per_model=2, timeout=150):
    """Try each model in order, a few times, backing off. Returns (text, model) or (None, None).
    timeout = idle seconds between streamed tokens, not total time."""
    for model in models:
        for attempt in range(tries_per_model):
            try:
                txt, fin = _stream(model, system, user, max_tokens, temperature, timeout)
                if fin == "length" or not txt.strip():
                    log(f"  {model} finish={fin} len={len(txt)} -> retry"); time.sleep(3); continue
                return txt, model
            except urllib.error.HTTPError as e:
                code = e.code
                log(f"  {model} HTTP {code} attempt {attempt+1}")
                if code in (400, 404, 410, 403):
                    break  # model not usable for us; next model
                time.sleep(min(60, 8 * (attempt + 1)) + random.random() * 3)
            except Exception as e:
                log(f"  {model} err {str(e)[:80]} attempt {attempt+1}"); time.sleep(8 * (attempt + 1))
    return None, None

# ---------------------------------------------------------------- chunking
def chunk_book(chapters):
    """Yield chunks: {id, chapter, title, paras:[(pid, text)], words}. Never splits a paragraph; never crosses chapters."""
    out = []
    for ci, ch in enumerate(chapters):
        cur, words = [], 0
        for pi, p in enumerate(ch["paras"]):
            w = len(p.split())
            if cur and words + w > CHUNK_WORDS:
                out.append({"id": f"c{ci:02d}_{len([o for o in out if o['chapter']==ci]):02d}", "chapter": ci,
                            "title": ch["title"], "paras": cur, "words": words}); cur, words = [], 0
            cur.append((pi, p)); words += w
        if cur:
            out.append({"id": f"c{ci:02d}_{len([o for o in out if o['chapter']==ci]):02d}", "chapter": ci,
                        "title": ch["title"], "paras": cur, "words": words})
    return out

def fmt_src(paras):
    return "\n\n".join(f"¶{pid} {text}" for pid, text in paras)

def parse_out(txt, paras):
    """Map ¶n markers back; returns dict pid->text and list of missing pids."""
    got = {}
    for m in re.finditer(r"¶\s?(\d+)\s*(.*?)(?=(?:\n\s*¶\s?\d+)|\Z)", txt, re.S):
        got[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    missing = [pid for pid, _ in paras if pid not in got or not got[pid]]
    return got, missing

# ---------------------------------------------------------------- prompts
def load_style(lang):
    return open(os.path.join(ROOT, "style_sheets", "_common.md")).read() + "\n\n" + \
           open(os.path.join(ROOT, "style_sheets", f"{lang}.md")).read()

def gloss_block(gl, text):
    present = [n for n in gl["characters"] if re.search(r"\b" + re.escape(n) + r"\b", text)]
    main = gl["characters"][:25]
    names = sorted(set(present) | set(main), key=lambda n: gl["characters"].index(n))
    return ("GLOSSARY — keep these spellings exactly, never translate, never inflect the name itself:\n"
            f"Characters: {', '.join(names)}\nPlaces/teams/brands (keep English): {', '.join(gl['places_teams_brands'])}\n"
            f"Family nicknames kept as-is: {', '.join(gl.get('family_nicknames_keep', []))}\n{gl.get('notes','')}\n"
            f"Rule: {gl['rule']}")

def translate_chunk(lang, style, gl, chunk, prev_tail):
    src = fmt_src(chunk["paras"])
    system = (f"You translate English commercial romance into {LANG_NAME[lang]}.\n\n{style}\n\n{gloss_block(gl, src)}")
    user = ""
    if prev_tail:
        user += f"CONTEXT — the end of the previous passage (English → your translation), for continuity only, do NOT re-translate:\n{prev_tail}\n\n"
    user += f"TRANSLATE the following passage from '{chunk['title']}'. Return every ¶ marker with its translated paragraph:\n\n{src}"
    for round_ in range(3):
        txt, model = chat(TRANSLATORS, system, user)
        if not txt:
            return None, None, ["no model answered"]
        got, missing = parse_out(txt, chunk["paras"])
        if not missing:
            return got, model, []
        log(f"  {chunk['id']} missing ¶ {missing[:8]} from {model}; retry {round_+1}")
        user = user + f"\n\nYour previous answer dropped these paragraph markers: {missing}. Return ALL markers, each with its full translation."
    return got, model, [f"missing paragraphs {missing}"]

# ---------------------------------------------------------------- Test A: lint
def lint_chapter(lang, gl, src_paras, tgt_paras):
    src = " ".join(src_paras); tgt = " ".join(tgt_paras)
    issues = []
    for n in gl["characters"]:
        a = len(re.findall(r"\b" + re.escape(n) + r"(?:['’]s)?\b", src)); b = len(re.findall(r"\b" + re.escape(n) + r"(?:s|['’]s|es|i|a)?\b", tgt))
        if a and abs(a - b) > max(1, a // 4):
            issues.append({"kind": "name_count", "name": n, "src": a, "tgt": b})
    ratio = len(tgt.split()) / max(1, len(src.split()))
    lo, hi = (0.85, 1.35) if lang == "de" else (0.9, 1.4)
    if not (lo <= ratio <= hi): issues.append({"kind": "length_ratio", "ratio": round(ratio, 2)})
    if len(src_paras) != len(tgt_paras): issues.append({"kind": "para_count", "src": len(src_paras), "tgt": len(tgt_paras)})
    if sum(p == "***" for p in src_paras) != sum(p == "***" for p in tgt_paras): issues.append({"kind": "scene_breaks"})
    if not QUOTE_OK[lang](tgt) and '"' in src: issues.append({"kind": "quote_style"})
    for pat in BANNED[lang]:
        for m in re.finditer(pat, tgt):
            issues.append({"kind": "banned", "match": tgt[max(0, m.start()-30):m.end()+30]})
    formal = len(re.findall(FORMAL[lang], tgt)); informal = len(re.findall(INFORMAL[lang], tgt, re.I))
    ital_src = src.count("*"); ital_tgt = tgt.count("*")
    if ital_src and abs(ital_src - ital_tgt) > max(2, ital_src // 3): issues.append({"kind": "italics", "src": ital_src, "tgt": ital_tgt})
    return {"issues": issues, "formal": formal, "informal": informal, "ratio": round(ratio, 2)}

# ---------------------------------------------------------------- Test B: judges
JUDGE_SYS = ("You are a native {L} speaker who reads a lot of commercial romance. You receive a passage in {L} ONLY — not the "
             "English original. Read it as a reader. Report in ENGLISH as JSON only: "
             '{{"overall_1to10": n, "reads_as_translation": true/false, '
             '"stumbles":[{{"quote":"exact words from the passage","problem":"what a native trips on","fix":"better {L}"}}], '
             '"register":"formal/informal address handling, and whether any switch felt right", '
             '"names_and_terms":"inconsistencies or oddities", "cadence":"does dialogue snap, do sentences vary, anything monotone"}}. '
             "Be specific and strict. Quote exactly. Empty stumbles only if truly clean.")

def judge_chunk(lang, translator_model, tgt_text):
    fam = lambda m: m.split("/")[0]
    pool = [m for m in JUDGES_POOL if fam(m) != fam(translator_model)]
    reports = []
    for i in range(2):
        cand = [m for m in pool if fam(m) not in {fam(r["judge"]) for r in reports}]
        txt, model = chat(cand, JUDGE_SYS.format(L=LANG_LABEL[lang]), tgt_text, max_tokens=6000, temperature=0.2, tries_per_model=2, timeout=600)
        if not txt: break
        m = re.search(r"\{.*\}", txt, re.S)
        try: j = json.loads(m.group(0)) if m else {}
        except Exception: j = {}
        if isinstance(j.get("overall_1to10"), (int, float)):
            reports.append({"judge": model, "report": j})
        else:
            log(f"  judge {model} returned no usable JSON ({len(txt)} chars) -> next model")
    return reports

def fix_lines(lang, style, gl, src_text, tgt_text, stumbles):
    system = f"You are a senior {LANG_NAME[lang]} romance editor.\n\n{style}\n\n{gloss_block(gl, src_text)}"
    user = ("Below is a passage (with ¶ markers) and a list of reader stumbles with suggested fixes. Apply ONLY these fixes "
            "(you may improve a suggested fix if it is wrong for the context), change nothing else, keep every ¶ marker, "
            "and return the full passage.\n\nSTUMBLES:\n" + json.dumps(stumbles, ensure_ascii=False, indent=1) +
            "\n\nPASSAGE:\n" + tgt_text)
    txt, model = chat(TRANSLATORS[1:] + TRANSLATORS[:1], system, user, temperature=0.2)
    return txt, model

# ---------------------------------------------------------------- honorific sweep (deterministic, after fixes)
HONORIFICS = {"de": [(r"\bMrs\.\s+", "Frau "), (r"\bMs\.\s+", "Frau "), (r"\bMr\.\s+", "Herr "), (r"\bMiss\s+(?=[A-Z])", "Frau ")],
              "pt-BR": [(r"\bMrs\.\s+", "Sra. "), (r"\bMs\.\s+", "Sra. "), (r"\bMr\.\s+", "Sr. "), (r"\bMiss\s+(?=[A-Z])", "Srta. ")],
              "es": [(r"\bMrs\.\s+", "Sra. "), (r"\bMs\.\s+", "Sra. "), (r"\bMr\.\s+", "Sr. "), (r"\bMiss\s+(?=[A-Z])", "Srta. ")],
              "it": [(r"\bMrs\.\s+", "signora "), (r"\bMs\.\s+", "signora "), (r"\bMr\.\s+", "signor "), (r"\bMiss\s+(?=[A-Z])", "signorina ")],
              "fr": [(r"\bMrs\.\s+", "Mme "), (r"\bMs\.\s+", "Mme "), (r"\bMr\.\s+", "M. "), (r"\bMiss\s+(?=[A-Z])", "Mlle ")]}
def sweep(lang, text):
    for pat, rep in HONORIFICS.get(lang, []): text = re.sub(pat, rep, text)
    return text

# ---------------------------------------------------------------- docx
def build_docx(lang, book, chapters, tgt_by_chapter, out_path):
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    kw, ep, pr = CHAPTER_WORD[lang]
    for ci, ch in enumerate(chapters):
        t = ch["title"]
        t = re.sub(r"^Chapter\s+", kw + " ", t, flags=re.I); t = re.sub(r"^Epilogue", ep, t, flags=re.I); t = re.sub(r"^Prologue", pr, t, flags=re.I)
        doc.add_heading(t, level=1)
        for p in tgt_by_chapter[ci]:
            if p == "***":
                para = doc.add_paragraph("* * *"); para.alignment = WD_ALIGN_PARAGRAPH.CENTER; continue
            para = doc.add_paragraph()
            for k, seg in enumerate(p.split("*")):
                if not seg: continue
                run = para.add_run(seg); run.italic = (k % 2 == 1)
    doc.save(out_path)

# ---------------------------------------------------------------- per book × lang
def run_book(series, book, lang, gl):
    outdir = os.path.join(ROOT, "books", series, book, lang); os.makedirs(outdir, exist_ok=True)
    summ_path = os.path.join(outdir, "summary.json")
    if os.path.exists(summ_path) and json.load(open(summ_path)).get("done"):
        log(f"SKIP {book} [{lang}] already done"); return json.load(open(summ_path))
    epub = sorted(glob.glob(os.path.join(SOURCE_DIR, book, "*.epub")))[0]
    chapters = extract(epub)
    chunks = chunk_book(chapters)
    style = load_style(lang)
    log(f"START {book} [{lang}] chapters={len(chapters)} chunks={len(chunks)} words={sum(c['words'] for c in chapters)}")
    # --- translate (resumable)
    cpath = os.path.join(outdir, "chunks.jsonl")
    done = {}
    if os.path.exists(cpath):
        for l in open(cpath):
            try: d = json.loads(l); done[d["id"]] = d
            except Exception: pass
    stats = {"chunks": len(chunks), "translated": 0, "failed": [], "models": {}, "lint_issues": 0, "judged": 0,
             "fixed": 0, "avg_score": None, "flagged_lines": 0, "unresolved": []}
    cf_lock = threading.Lock()
    # translation runs in chapter order within a chapter (context tail), chapters in parallel
    by_ch = {}
    for c in chunks: by_ch.setdefault(c["chapter"], []).append(c)
    def do_chapter(ci):
        prev_tail = ""
        for c in by_ch[ci]:
            if c["id"] in done and not done[c["id"]].get("errors"):
                d = done[c["id"]]
            else:
                got, model, errs = translate_chunk(lang, style, gl, c, prev_tail)
                d = {"id": c["id"], "chapter": ci, "model": model, "errors": errs,
                     "tgt": [[pid, got.get(pid, "") if got else ""] for pid, _ in c["paras"]]}
                with cf_lock:
                    with open(cpath, "a") as f: f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    done[c["id"]] = d
            if d["tgt"]:
                last_src = c["paras"][-1][1][-400:]; last_tgt = (d["tgt"][-1][1] or "")[-400:]
                prev_tail = f"EN: …{last_src}\n{LANG_LABEL[lang]}: …{last_tgt}"
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(do_chapter, sorted(by_ch)))
    for c in chunks:
        d = done.get(c["id"], {})
        if d.get("errors"): stats["failed"].append(c["id"])
        else:
            stats["translated"] += 1; stats["models"][d.get("model")] = stats["models"].get(d.get("model"), 0) + 1
    # --- assemble per chapter, lint
    tgt_by_chapter = {}
    for ci, ch in enumerate(chapters):
        t = []
        for c in by_ch.get(ci, []):
            d = done.get(c["id"], {"tgt": []}); m = {pid: txt for pid, txt in d.get("tgt", [])}
            for pid, src in c["paras"]:
                t.append("***" if src == "***" else (m.get(pid) or f"[[UNTRANSLATED ¶{pid}]] " + src))
        tgt_by_chapter[ci] = t
    lint = {ci: lint_chapter(lang, gl, ch["paras"], tgt_by_chapter[ci]) for ci, ch in enumerate(chapters)}
    stats["lint_issues"] = sum(len(v["issues"]) for v in lint.values())
    json.dump(lint, open(os.path.join(outdir, "lint.json"), "w"), ensure_ascii=False, indent=1)
    # --- judges per chunk, then fixes
    jpath = os.path.join(outdir, "judge.jsonl"); fpath = os.path.join(outdir, "fixes.jsonl")
    judged = {}
    if os.path.exists(jpath):
        for l in open(jpath):
            try:
                j = json.loads(l)
                if j.get("reports"): judged[j["id"]] = j   # empty judgements are redone, never treated as done
            except Exception: pass
    def do_judge(c):
        if c["id"] in judged: return judged[c["id"]]
        d = done.get(c["id"]);
        if not d or d.get("errors"): return None
        tgt_text = "\n\n".join(f"¶{pid} {txt}" for pid, txt in d["tgt"])
        reps = judge_chunk(lang, d.get("model") or "", tgt_text)
        if not reps:
            log(f"  {c['id']} no judge answered; left for next pass"); return None
        j = {"id": c["id"], "reports": reps}
        with cf_lock:
            with open(jpath, "a") as f: f.write(json.dumps(j, ensure_ascii=False) + "\n")
            judged[c["id"]] = j
        return j
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(do_judge, chunks))
    scores = []
    for c in chunks:
        j = judged.get(c["id"])
        if not j: continue
        stats["judged"] += 1
        for r in j["reports"]:
            s = r["report"].get("overall_1to10")
            if isinstance(s, (int, float)): scores.append(s)
        stumbles = []
        for r in j["reports"]:
            for s in r["report"].get("stumbles", []) or []:
                if isinstance(s, dict) and s.get("quote"): stumbles.append(s)
        # dedupe by quote
        seen = set(); uniq = []
        for s in stumbles:
            k = s["quote"].strip().lower()
            if k not in seen: seen.add(k); uniq.append(s)
        # also add lint banned matches for this chapter as stumbles
        for iss in lint[c["chapter"]]["issues"]:
            if iss["kind"] == "banned": uniq.append({"quote": iss["match"], "problem": "banned form (style sheet)", "fix": "localize per style sheet"})
        if not uniq: continue
        stats["flagged_lines"] += len(uniq)
        d = done[c["id"]]
        if d.get("fixed_by"): stats["fixed"] += 1; continue   # already fixed before a restart
        tgt_text = "\n\n".join(f"¶{pid} {txt}" for pid, txt in d["tgt"])
        src_text = fmt_src(c["paras"])
        fixed, model = fix_lines(lang, style, gl, src_text, tgt_text, uniq)
        if fixed:
            got, missing = parse_out(fixed, c["paras"])
            if not missing:
                d["tgt"] = [[pid, got[pid]] for pid, _ in c["paras"]]; d["fixed_by"] = model; stats["fixed"] += 1
                with open(fpath, "a") as f: f.write(json.dumps({"id": c["id"], "stumbles": uniq, "model": model}, ensure_ascii=False) + "\n")
                with open(cpath, "w") as f:   # persist the fixed text now, so a restart never loses fix work
                    for cc in chunks:
                        if cc["id"] in done: f.write(json.dumps(done[cc["id"]], ensure_ascii=False) + "\n")
            else:
                stats["unresolved"].append({"id": c["id"], "reason": f"fix dropped ¶ {missing[:5]}"})
        else:
            stats["unresolved"].append({"id": c["id"], "reason": "no fixer answered"})
    stats["avg_score"] = round(sum(scores) / len(scores), 2) if scores else None
    # deterministic honorific sweep on every paragraph
    for c in chunks:
        d = done.get(c["id"])
        if d and d.get("tgt"): d["tgt"] = [[pid, sweep(lang, txt or "")] for pid, txt in d["tgt"]]
    # re-judge a sample of fixed chunks so every book reports an AFTER score
    after = []
    sample = [c for c in chunks if c["id"] in {j["id"] for j in judged.values()}][:: max(1, len(chunks) // 5)][:5]
    for c in sample:
        d = done[c["id"]]; tgt_text = "\n\n".join(f"¶{pid} {txt}" for pid, txt in d["tgt"])
        for r in judge_chunk(lang, d.get("model") or "", tgt_text):
            sc = r["report"].get("overall_1to10")
            if isinstance(sc, (int, float)): after.append(sc)
    stats["avg_score_after_fixes_sample"] = round(sum(after) / len(after), 2) if after else None
    # rewrite chunks.jsonl with fixes applied, re-assemble, re-lint, build docx
    with open(cpath, "w") as f:
        for c in chunks:
            if c["id"] in done: f.write(json.dumps(done[c["id"]], ensure_ascii=False) + "\n")
    for ci, ch in enumerate(chapters):
        t = []
        for c in by_ch.get(ci, []):
            d = done.get(c["id"], {"tgt": []}); m = {pid: txt for pid, txt in d.get("tgt", [])}
            for pid, src in c["paras"]:
                t.append("***" if src == "***" else (m.get(pid) or f"[[UNTRANSLATED ¶{pid}]] " + src))
        tgt_by_chapter[ci] = t
    lint2 = {ci: lint_chapter(lang, gl, ch["paras"], tgt_by_chapter[ci]) for ci, ch in enumerate(chapters)}
    json.dump(lint2, open(os.path.join(outdir, "lint_after_fixes.json"), "w"), ensure_ascii=False, indent=1)
    stats["lint_issues_after"] = sum(len(v["issues"]) for v in lint2.values())
    docx_path = os.path.join(outdir, f"{book}-{LANG_LABEL[lang]}.docx")
    build_docx(lang, book, chapters, tgt_by_chapter, docx_path)
    stats.update({"book": book, "lang": lang, "docx": docx_path, "done": not stats["failed"],
                  "finished": datetime.now().isoformat(timespec="minutes")})
    json.dump(stats, open(summ_path, "w"), ensure_ascii=False, indent=1)
    log(f"END {book} [{lang}] translated={stats['translated']}/{stats['chunks']} failed={len(stats['failed'])} "
        f"lint={stats['lint_issues']}->{stats['lint_issues_after']} judged={stats['judged']} avg={stats['avg_score']} "
        f"flagged={stats['flagged_lines']} fixed={stats['fixed']} unresolved={len(stats['unresolved'])}")
    return stats

# ---------------------------------------------------------------- index
def write_index(series):
    rows = []
    for summ in sorted(glob.glob(os.path.join(ROOT, "books", series, "*", "*", "summary.json"))):
        s = json.load(open(summ))
        rows.append(f"- {s['book']} [{s['lang']}] — {'DONE' if s.get('done') else 'INCOMPLETE'} · chunks {s['translated']}/{s['chunks']} · "
                    f"judge avg {s.get('avg_score')} · flagged {s.get('flagged_lines')} fixed {s.get('fixed')} · lint {s.get('lint_issues')}→{s.get('lint_issues_after')} · "
                    f"unresolved {len(s.get('unresolved', []))} · {os.path.basename(s['docx'])} · {s['finished']}")
    idx = os.path.join(ROOT, "INDEX.md")
    head = "# Translation status board\n\nOne line per book × language. Covers/metadata/uploads tracked below as they happen.\n\n## Translations\n"
    open(idx, "w").write(head + "\n".join(rows) + "\n")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--series", required=True); ap.add_argument("--langs", default="de")
    ap.add_argument("--books", default="all"); ap.add_argument("--order", default=None, help="json list file of book folder names in order")
    a = ap.parse_args()
    gl = json.load(open(os.path.join(ROOT, "glossaries", f"{a.series}.json")))
    order = json.load(open(a.order or os.path.join(ROOT, "glossaries", f"{a.series}_books.json")))
    books = order if a.books == "all" else [b.strip() for b in a.books.split(",")]
    if a.cmd == "status":
        write_index(a.series); print(open(os.path.join(ROOT, "INDEX.md")).read()); return
    langs = [l.strip() for l in a.langs.split(",")]
    for lang in langs:            # one language at a time, all books, then next language
        for book in books:
            try:
                run_book(a.series, book, lang, gl)
            except Exception as e:
                log(f"ERROR {book} [{lang}] {e!r}")
            write_index(a.series)
    log("ALL DONE")

if __name__ == "__main__":
    main()
