# Research Paper Navigator

Final capstone for the MIT xPRO RAG and Context Engineering course. The navigator answers questions about research-paper PDFs, using keyword and semantic retrieval, a bounded retrieval agent, and citations to the selected paper pages.

The original collection contains **153 PDFs and 12,420 overlapping chunks**. The final system carries forward the tested Week 2 hybrid retriever, Week 5 retrieval functions, and Week 7 safeguards. A portable command-line interface adds ordinary question answering, recorded demonstrations, and reproducible evidence summaries.

## Install

Python 3.10 or newer is required. From this repository directory:

```sh
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Review without an API key or PDFs

```sh
python navigator.py check
python -m unittest discover -s tests -v
python navigator.py summary
python navigator.py replay --case P1 --arm hardened
python navigator.py replay --case S1 --arm baseline
python navigator.py replay --case S1 --arm hardened
```

`replay` displays saved Week 7 model wording, clearly labeled as a recorded demonstration. It does not generate a new answer. The tests use small synthetic chunks and mocked provider replies to exercise the actual direct path, hybrid rank fusion, named-paper agent, integrity checks, and cache behavior. These are implementation checks, not accuracy benchmarks.

## Configure live questions

Two demonstration papers are included in `data/sample_papers/` in this private course-submission repository. The CLI uses them by default. To use the complete collection, put your authorized PDF copies directly inside `data/papers/`, or pass `--corpus` with another folder. The full collection and embedding caches are excluded from version control. `data/corpus_manifest.json` lists the original collection. Historical experiment scores refer to all 153 PDFs, not the two-paper sample.

Set the key in your process environment. `.env.example` documents variable names but the program does not automatically load it.

```powershell
# Windows PowerShell. Supply your own key locally.
$env:OPENROUTER_API_KEY = '<your key>'
$env:RPN_CORPUS_DIR = 'C:\path\to\ResearchPapers'
```

```sh
# macOS/Linux
export OPENROUTER_API_KEY='<your key>'
export RPN_CORPUS_DIR='/path/to/ResearchPapers'
```

The default chat model is `openai/gpt-5.4-mini`. Override it with `--model` or `RPN_CHAT_MODEL`. Embeddings use `openai/text-embedding-3-small` at 512 dimensions. Your OpenRouter account must support the configured models and have sufficient credits. Changing a chat model also changes the answer-cache key.

```sh
python navigator.py ask "How does Learning to Infer Program Sketches combine a neural model with symbolic program synthesis?" --save outputs/method.json
python navigator.py ask "Compare Learning to Infer Program Sketches with Program Synthesis with a REPL. How does each combine learning with symbolic search or execution feedback? For each paper, report one evaluation or scope trade-off stated by its authors." --save outputs/comparison.json
python navigator.py keyword "Learning to Infer Program Sketches symbolic search"
```

The first live run extracts PDF text and embeds the corpus, which can take several minutes and incur API charges. Later runs reuse `.cache/`. Live mode sends embedding inputs and selected excerpts to OpenRouter. `keyword` searches locally with BM25 and does not generate an LLM answer. `--no-cache` forces a new live answer. `--cache-dir` selects a separate cache location.

## How the final workflow works

1. Extract text page by page with pypdf. Split pages into 350-word chunks with 70-word overlap, preserving the PDF filename and page.
2. Rank chunks with BM25 and cosine similarity. Combine the two ranks with equal-weight reciprocal rank fusion, with rank constant 60. Limit repeated passages from a PDF.
3. Remove recognizable role commands and pasted fake-source blocks. Escape tagged prompt fields and treat retrieved text as evidence, never as instructions.
4. Route simple questions to one search and one answer. Comparison, evaluation, scope, limitation, and benchmark questions use a bounded agent. At most three retrieval actions are available: one initial search and up to two further searches or paper lookups.
5. Derive explicitly named papers from the question and corpus titles, rather than supplying evaluation answer labels to the live system. Focused lookups and selection reserve evidence for up to two named papers.
6. Compare selected text with the loaded index copy. Cap answer context at 18,000 characters and each passage at 4,400. Generate an answer with numbered citations and source pages.
7. Reuse a successful direct answer for an identical normalized, self-contained question within seven days, under the same corpus, model, and policy. The final CLI retains source hits with cached wording. Paraphrases miss this exact cache.

This is a sequential local CLI, not a concurrent web service. Historical experiments used known title anchors in their test harness. The final CLI derives anchors from the actual query. The final wrapper extends the Week 7 harness cache to ordinary direct questions and preserves their source metadata. Historical measurements still refer to the original Week 7 experiment, not a new benchmark of the wrapper.

## Evaluation evidence

All JSON files in `evidence/` preserve saved experimental results. Local course paths in location metadata are changed to relative `course/` references. Generated answer strings remain unchanged. These references identify earlier course artifacts and are not imports required by the final code.

| Experiment | Observed result | Interpretation |
| --- | --- | --- |
| Week 3: eight questions | Expected source in top five for 7/7 answerable cases. Judge passed 8/8, but combined acceptance was 7/8. | An independent abstention guard caught an unsupported identification the judge accepted. |
| Week 4: six broad questions, paired retrieval paths | Both paths passed 6/6 rubric checks. Distinct filename slots increased from 23 to 27 with decomposition. | Wider coverage did not establish better answer accuracy. Duplicate versions can inflate diversity. |
| Week 5: three multistep questions | Fixed search and initial agent passed 2/3. Refined agent passed 3/3. | Focused evidence selection helped on this small set. |
| Week 6: model-choice audit | Current and mixed setups passed 3/3 ordinary questions. All-cheaper setup passed 0/3. | Lower provider cost alone did not justify changing both models. |
| Week 7: seven paired cases | 68,687 to 40,115 production tokens, 76.004 to 38.094 observed seconds, $0.07200673 to $0.044558 reported cost. | 41.6% fewer tokens and 38.1% lower reported cost in one small sequential run. |
| Week 7: exact repeat | Identical saved answer with zero API calls. | Local cache savings apply only to eligible repeats. |

Different weeks used different questions and rubrics. Do not treat these rows as one comparable accuracy trend. Week 7 totals exclude corpus index construction, the cache repeat, and superseded iterations. Provider timing and pricing vary. Week 7 has no independent source-aware judge score.

The final Week 8 interface was also checked with a live method question, a live two-paper comparison, and an exact repeat. The repeat used zero API calls. The comparison exposed an extracted-title suffix issue, which was corrected and covered by a regression test. It also retained a claim-to-citation mismatch, documented in `evidence/final_interface_review.txt`. These are functional checks, not an independently graded accuracy benchmark. Full generated wording and source hits are preserved in the separate `final_interface_*.json` files.

## Known limitations

- PDF extraction can flatten tables and misalign column labels. A final Week 7 answer rejected an invented statistic but still mixed full-dataset and filtered-test results. Numeric claims require inspection of the actual paper table.
- Regex filters miss novel attacks and may remove legitimate quoted instructions. Prompt tags do not guarantee attack resistance.
- Integrity checks reject text changed after ingestion. A misleading or poisoned PDF already present at ingestion can still match the stored text.
- Conference and preprint duplicates remain. File diversity does not equal distinct studies.
- Exact title detection is conservative and can miss shortened names. The action cap supports at most two focused named-paper lookups. Broad questions may still have gaps.
- Cached answers reuse their original quality. There is no independent automatic factual validation before caching. Disable caching when checking revised findings. PDF updates are observed when the corpus is reloaded, not by a background watcher.
- Extraction caching uses filenames, file size, and modification time. A content change that preserves all of those could evade cache invalidation. Delete `.cache/` locally to force reconstruction after such a change.
- The small test sets and shared answer/judge model have blind spots. There is no multi-turn memory or production load evaluation.

## Project files

- `navigator.py`: supported command-line entry point.
- `src/research_navigator/`: hybrid retrieval, bounded agent, safeguards, and portable interface.
- `tests/`: deterministic offline integration checks.
- `evidence/week2_results.json` through `week7_results.json`: historical traces, questions, outputs, and measurements.
- `evidence/week7_answers_verbatim.log`: all final paired answers in full.
- `data/corpus_manifest.json`: original PDF inventory. `data/sample_papers/` contains the two demonstration inputs.
- `presentation/`: final deck and narration once exported.

The final presentation explains the problem, workflow, evidence, decisions, and trade-offs. The assignment also requires a separate 8–10 minute recording and a GitHub link that the Learning Facilitator can access. Repository publication does not upload a recording or submit the assignment.
