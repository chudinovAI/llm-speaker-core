# Project State

## Goal

`llm-speaker-core` is a voice-first local assistant for GUAP. The current target is:

- high factual precision on official GUAP queries;
- fast local runtime on modest hardware;
- low memory pressure in the default path;
- optional heavier retrieval components built offline on stronger hardware.

## Current Architecture

### Runtime

- `src/llm_speaker_core/voice/`
  - ASR pipeline
  - bridge from transcript to service
  - TTS
  - one-command `llm-voice-stack`
- `src/llm_speaker_core/app/bootstrap.py`
  - builds the service graph
- `src/llm_speaker_core/service.py`
  - answer orchestration
  - grounding guard
  - speaker/display shaping
- `src/llm_speaker_core/llm_runtime/client.py`
  - Ollama chat client

### Retrieval

- `src/llm_speaker_core/ingest/`
  - Cloudflare crawl ingestion
  - manual document ingestion
  - normalization
  - structure-aware chunking
  - document quality flags
  - OCR fallback for PDFs
- `src/llm_speaker_core/retrieval/`
  - BM25 lexical index
  - optional dense FAISS index
  - optional reranker
  - hybrid retrieval service
  - gold-set evaluation

## Active Data Sources

### Canonical corpus inputs

- `data/raw/cloudflare/latest/records.jsonl`
- `data/raw/cloudflare/latest/seeds/*`
- `data/raw/manual_docs/pr_guap_05-145-25.pdf`

### Planned official seed coverage

Current Cloudflare seed set:

- `https://guap.ru`
- `https://priem.guap.ru`
- `https://guap.ru/sveden`
- `https://guap.ru/eif`
- `https://guap.ru/studlife`
- `https://guap.ru/vrmp`

### Removed from the canonical pipeline

- old `firecrawl` raw dump
- operational/faculty side branches that were not part of the current official corpus strategy

## Current Artifacts

Built locally on `2026-03-31`:

- raw web records: `271`
- normalized documents: `272`
- indexed chunks: `680`
- index manifest version: `hybrid-rag-v3`

Current manifest:

- `data/index_manifest.json`
- `doc_count = 187`
- `chunk_count = 680`
- `dense_available = false`
- `reranker_available = true`

Notes:

- normalized documents are larger than indexed documents because quality filtering and dedup happen before indexing;
- current local baseline was evaluated in fast mode with `dense=False` and `reranker=False`, because this is the relevant low-memory local operating mode.

## Current Metrics

### Local fast baseline

Source:

- `data/eval/hybrid_eval_report.json`

Metrics on `data/eval/gold_queries.jsonl`:

- queries: `15`
- recall@k: `0.8667`
- top1 hit rate: `0.8`
- top3 hit rate: `0.8667`
- MRR: `0.8333`

### Queries that still fail or underperform

1. `Какие сроки приема в ГУАП?`
- expected: `.../bach/dates`
- actual retrieval still prefers `.../rules`

2. `Как связаться с приемной комиссией ГУАП?`
- expected: `priem.guap.ru/contacts`
- actual retrieval still drifts to generic admission pages like `bach`, `calc`, `spo/calc`, and sometimes `sveden/common`

3. `Как оплатить обучение в ГУАП?`
- recall is acceptable, but top1 is still wrong
- payment queries still lean toward tuition listings instead of payment-specific pages

## What Was Wrong With The Previous State

### Removed legacy

This cleanup removed the following obsolete or harmful pieces:

- old `firecrawl` ingestion path and raw dataset
- unused `llm.py` compatibility wrapper
- operational retrieval branches for:
  - `library`
  - `hr`
  - `medical`
  - `support`
  - `faculty_contacts`
- tests that only covered those removed branches
- generated transport archives and local runtime artifacts
- caches and `egg-info`

### Why this mattered

These branches created three concrete problems:

1. retrieval complexity grew without helping the active corpus;
2. ranking priors were partially optimized for sources that are no longer canonical;
3. memory and maintenance cost increased for features that are outside the current product focus.

## Current RAG Assessment

### What is good now

- official GUAP corpus is now canonical and narrow;
- directions, tuition price, location, student life, student unions, official info, budget, dorm and VRMP are already in a workable range;
- lexical baseline with facet priors is strong enough to be the default runtime path;
- project state is cleaner and closer to a replaceable retrieval core.

### What is still wrong

1. facet separation inside admission is incomplete
- `dates` still loses to `rules`
- `contacts` still loses to generic admission hubs and calculators

2. tuition action queries are not separated enough
- `payment` is still too close to `price` and `listing`

3. grounding score is still too optimistic
- semantically close but wrong pages can keep a high score
- score calibration is not yet strict enough to be treated as final confidence

4. dense retrieval is not the default local answer
- this is correct for low-memory hardware
- but it means the local fast path must stay strong on lexical + metadata priors alone

## Implementation Plan

The plan below is intentionally shaped by the constraints:

- fast runtime
- low memory
- high precision

### Phase 1. Stabilize the fast baseline

Keep the default local runtime:

- BM25 lexical retrieval as primary candidate generator
- no dense by default
- reranker disabled by default on weak hardware

Work:

1. strengthen facet priors for `admission_dates`
2. strengthen facet priors for `admission_contacts`
3. strengthen facet priors for `tuition_payment`
4. reduce generic admission page dominance

Target:

- improve top1 without increasing runtime cost materially

### Phase 2. Recalibrate ranking, not answer hacks

Work:

1. add stronger path penalties for:
- `/rules` when query facet is `admission_dates`
- `/calc`, `/spo/calc`, `/bach` when query facet is `admission_contacts`
- listing pages when query facet is `tuition_payment`

2. add stronger positive priors for:
- `priem.guap.ru/bach/dates`
- `priem.guap.ru/contacts`
- `/eif/pay`
- `/eif/pol_usl`
- `/eif/inf_dog`

3. keep changes in retrieval metadata and scoring only
- do not patch answer generation for these cases

### Phase 3. Tighten grounding score

Work:

1. include facet agreement more strongly in grounding score
2. include page-type mismatch as a negative signal
3. include top1 vs top2 score margin
4. lower confidence when top result is a generic hub for a narrow query

Target:

- `grounding_score` should better reflect whether the top hit is actually answer-worthy

### Phase 4. Optional heavy path on GPU hardware

For Kaggle/Colab or stronger local hardware:

1. rebuild dense artifacts with `BAAI/bge-m3`
2. enable reranking with `BAAI/bge-reranker-v2-m3`
3. keep candidate pool small before rerank
- lexical top-N first
- dense optional union
- rerank only the compact merged set

Constraint:

- dense/rerank must stay optional and never become required for the default laptop path

### Phase 5. Evaluation discipline

Work:

1. keep the gold set small and clean
2. expand only with real user-important GUAP queries
3. track:
- recall@k
- top1 hit rate
- top3 hit rate
- MRR
- failure clusters by facet

## Immediate Next Work

1. fix `admission_dates`
2. fix `admission_contacts`
3. fix `tuition_payment` top1
4. recalibrate `grounding_score`

That is the shortest path to a better RAG under the current speed and memory limits.
