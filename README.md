# translate-novels

Translate your own novels from English into German, Brazilian Portuguese, Spanish, Italian and French — on free
models, with a native-reader quality pass, producing a Word file ready for Vellum (or any formatter).

No paid services. You bring your own free NVIDIA NIM API keys (build.nvidia.com); the pipeline uses the free tier
(rate-limited, ~40 requests/min per key) and streams every call so long chapters don't time out.

## What it does, per book × language
1. Extracts chapters from your EPUB (Vellum-built EPUBs work out of the box; scene breaks and italics are kept).
2. Chunks at paragraph boundaries (~1,000 words), never across a chapter.
3. Translates each chunk with a model ladder (gemma-4-31b → kimi-k3 → deepseek → glm), with your style sheet and
   glossary in every call. Paragraph markers guarantee nothing is dropped or added.
4. Lint (Test A): name counts vs the source, formal/informal address counts, quotation-mark style, length ratio,
   scene breaks, italics, banned forms (e.g. "Ms." left in German).
5. Two judges from *other* model families read the translation cold — no English — as native romance readers and
   report stumbles, register and cadence in English (Test B).
6. Flagged lines go back to the translator as line edits only; a deterministic honorific sweep follows; five chunks
   are re-judged so every book reports a before/after score.
7. Output: `books/<series>/<book>/<lang>/<book>-<Language>.docx` with Heading 1 chapters — import straight into Vellum.
Everything is checkpointed per chunk and resumable: re-run the same command after any interruption.

## Setup
```
git clone <this repo> && cd translate-novels
cp .env.example .env            # add your own NVIDIA keys
pip install python-docx         # the only non-stdlib dependency
mkdir -p source/"My First Novel" && cp /path/to/my-first-novel.epub source/"My First Novel"/
```
Put one folder per book under `source/` (or set `SOURCE_DIR=/path` in the environment).

## Glossary (names that must never change)
```
python3 build_glossary.py myseries      # needs glossaries/myseries_candidates.json — see build_glossary.py
```
or write `glossaries/myseries.json` by hand from `glossaries/example.json`, and list the books in reading order in
`glossaries/myseries_books.json` (folder names under `source/`).

## Run
```
python3 pipeline.py run --series myseries --langs de,pt-BR,es,it,fr --books all
python3 pipeline.py status --series myseries       # prints INDEX.md: one line per book × language
```
For an overnight run, detach it: `nohup caffeinate -i python3 pipeline.py run --series myseries --langs de --books all > logs/run.log 2>&1 &`

## Style sheets
`style_sheets/_common.md` holds the rules that apply to every language (faithfulness, names, heat level, idioms,
honorifics, measures, italics, output format). `de.md`, `pt-BR.md`, `es.md`, `it.md`, `fr.md` hold each language's
address rules (Sie/du, tú/usted, tu/Lei, tu/vous, você), punctuation, pet names, sports terms and chapter words.
Edit them for your books; they are prompts, not code.

## Honest notes
- Free-tier speed: roughly 40 minutes to translate a 45k-word novel with 6 parallel streams, then 2–4 hours of judging
  and fixing, depending on the lane's load. Reasoning models are called with thinking disabled; the ladder skips any
  model that returns nothing.
- Quality: first drafts scored ~6/10 with native-reader judges in our runs; the line-fix pass raises the re-judged
  sample by about half a point. Real native readers are still the last check before you publish.
- The judges and the translator are always different model families, so a model never grades itself.

## License
MIT
