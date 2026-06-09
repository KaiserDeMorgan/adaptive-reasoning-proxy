# Project Context: Adaptive Reasoning Proxy

## Who I am & what I'm building

I'm a Machine Learning Engineer building a standout resume project based on the
EDRM paper (arXiv:2605.22873 — "When Do LLMs Need to Think? A Dynamical Systems
View via Entropy Phase Transitions"). The goal is a production-grade, deployable
system that demonstrates deep LLM infrastructure knowledge.

The project is called the **Adaptive Reasoning Proxy** — a drop-in FastAPI proxy
that wraps any OpenAI-compatible API, intercepts the token stream, computes
Shannon entropy over logprobs in real time, and stops generation early when the
model has entered a low-entropy "settled reasoning regime". This cuts completion
tokens by an estimated 41–55% with under 2% accuracy loss on factual tasks.

---

## The core insight (from the paper)

Chain-of-thought reasoning is not always beneficial. The model passes through
distinct entropy phases during generation:

- **High entropy phase** = the model is actively exploring / reasoning. Let it run.
- **Low entropy phase** = the model has converged to a confident regime. Stop here.

We detect this transition using Shannon entropy H(t) = -∑ p log p computed over
the top-k logprob distribution at each token, smoothed with an exponential moving
average (EMA, alpha=0.15) over a rolling 20-token window.

---

## Project structure

```
adaptive-reasoning-proxy/
├── proxy/
│   ├── main.py              # FastAPI app — drop-in /v1/chat/completions endpoint
│   ├── entropy_engine.py        # EntropyState dataclass — H(t), EMA, should_stop()
│   ├── classifier.py            # Regex task router (FACTUAL/REASONING/CODE/CREATIVE)
│   ├── embedding_classifier.py  # Embedding nearest-centroid router (regex fallback)
│   ├── entropy_profiles.py      # DTW entropy-profile matching
│   ├── threshold_store.py       # Redis async store (+ in-memory fallback)
│   ├── query_log.py             # Async SQLite query log + stats aggregation
│   └── Dockerfile
├── dashboard/               # Next.js + Recharts dashboard (built)
├── tests/                   # pytest suite mirroring proxy/
├── docker-compose.yml       # proxy + redis + dashboard
├── pytest.ini
├── requirements.txt
├── .env.example
└── CLAUDE.md
```

---

## What's already built

All four Python files are complete and working:

**entropy_engine.py** — `EntropyState` dataclass with:
- `update(logprobs)` → computes H(t), updates EMA and rolling window
- `should_stop(threshold, min_tokens=40)` → returns True when recent_mean and
  EMA both fall below threshold (with min 40-token warmup, min 10-token window)

**classifier.py** — regex-based task router:
- `TaskType` enum: FACTUAL, REASONING, CREATIVE, CODE
- `THRESHOLDS` dict: {FACTUAL: 0.35, CREATIVE: 0.55, CODE: 0.60, REASONING: 0.75}
- `classify(prompt)` → returns TaskType

**threshold_store.py** — Redis async store:
- `get_threshold(task_type, default)` → reads from Redis or falls back to default
- `record_outcome(task_type, entropy_at_stop, user_rating)` → nudges threshold
  by ±0.02 based on user feedback (1=good, -1=truncated too early), logs to Redis

**main.py** — FastAPI proxy:
- `GET /health` → status check
- `POST /v1/chat/completions` → intercepts body, injects logprobs=True/top_logprobs=5,
  streams from OpenAI, runs entropy check per token, injects `edrm_meta` on early stop
- Early stop emits: `{"choices":[{"delta":{},"finish_reason":"entropy_stop"}], "edrm_meta": {...}}`

---

## Roadmap status — all four priorities complete

### Priority 1 — Tests ✅
`tests/` mirrors `proxy/` with a pytest suite (56 tests): entropy engine update
& should_stop edge cases, classifier per TaskType, the FastAPI endpoint via
`httpx.AsyncClient` with a mocked OpenAI stream (early-stop + natural finish),
DTW profiles, query log/stats, and embedding-classifier fallback. Run: `pytest`.

### Priority 2 — Dashboard (Next.js) ✅
Real-time dashboard in `dashboard/` (Next.js 14 + Recharts), polling the proxy
every 2s. 4 metric cards, entropy trace chart (H(t) + EMA with threshold and
stop markers), live threshold sliders, and a 50-row query log table. Reads from
the proxy's data API (see below). Needs Node.js to run (`npm install && npm run dev`).
Note: uses a zero-setup **SQLite** query log in the proxy as the stand-in for the
originally-planned Postgres table.

### Priority 3 — Embedding-based classifier ✅
`embedding_classifier.py` — `classify_async()` embeds the prompt and assigns it
to the nearest per-task **centroid** by cosine similarity, built from seed
prompts (expand toward ~500/class for production). Gated by `EMBEDDINGS_ENABLED`
and falls back to the regex classifier when disabled or the endpoint is down.

### Priority 4 — Entropy profile matching ✅
`entropy_profiles.py` — learns per-task entropy curve shapes online and matches
new curves with **DTW** distance (`dtw_distance`, `best_match`, `classify_by_profile`).
Currently an enrichment layer (exposed at `GET /profiles`); does not yet drive the
stop decision, so it cannot destabilise generation.

## Proxy data API (added for the dashboard)

- `GET /stats` — tokens saved, avg latency, stop rate, cost saved (estimates)
- `GET /logs?limit=N` — recent requests
- `GET /trace/latest` — last request's per-token H(t) + EMA + stop index
- `GET /thresholds` · `POST /thresholds/{task}` — read / set per-task thresholds
- `POST /feedback` — record a rating to nudge a threshold
- `GET /profiles` — learned DTW reference curves per task

---

## Tech stack

| Layer | Choice |
|---|---|
| Proxy server | FastAPI + uvicorn |
| LLM backend | OpenAI AsyncClient (drop-in for any OpenAI-compatible API) |
| Threshold store | Redis (redis.asyncio) |
| Dashboard | Next.js + Recharts |
| Infra | Docker Compose locally, Fly.io for deployment |
| Testing | pytest + pytest-asyncio + httpx |
| Python | 3.11 |

---

## Environment

Copy `.env.example` to `.env` and set:
```
OPENAI_API_KEY=sk-...
REDIS_URL=redis://redis:6379
```

Run locally:
```bash
docker-compose up
# or without Docker:
pip install -r requirements.txt
cd proxy && uvicorn main:app --reload
```

---

## Resume narrative (the "so what")

"Built a drop-in LLM inference proxy that dynamically detects when a model has
finished reasoning using entropy-phase analysis, cutting completion tokens by ~47%
with under 2% accuracy loss on factual benchmarks — saving meaningful API costs
at scale. Implemented adaptive per-task thresholds with a Redis feedback loop and
a real-time entropy trace dashboard."

The three numbers to measure and report:
- Token reduction: compare `usage.completion_tokens` proxy vs. direct on 200 queries
- Accuracy delta: run MMLU factual subset both ways, compare
- Latency overhead: should be under 15ms for the entropy computation

---

## Coding preferences

- Type hints on all functions
- Docstrings on all public methods
- Async throughout (no sync Redis or HTTP calls in the hot path)
- No print statements — use Python `logging` module
- Keep files small and single-responsibility
- Tests live in `tests/` at the project root, mirroring the `proxy/` structure
