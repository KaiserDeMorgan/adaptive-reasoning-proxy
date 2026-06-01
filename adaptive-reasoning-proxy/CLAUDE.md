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
│   ├── entropy_engine.py    # EntropyState dataclass — H(t), EMA, should_stop()
│   ├── classifier.py        # Regex task router (FACTUAL/REASONING/CODE/CREATIVE)
│   ├── threshold_store.py   # Redis async store — adaptive threshold updates
│   └── Dockerfile
├── dashboard/               # Next.js dashboard (not yet built — next priority)
├── docker-compose.yml       # proxy + redis + dashboard
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

## What needs to be built next

### Priority 1 — Tests
Write pytest tests for:
- `EntropyState.update()` with mock logprob dicts
- `EntropyState.should_stop()` edge cases (below min_tokens, window too small, threshold crossing)
- `classify()` for each TaskType
- FastAPI endpoint with `httpx.AsyncClient` test client (mock the OpenAI stream)

### Priority 2 — Dashboard (Next.js)
A real-time dashboard at `localhost:3000` reading from Redis + a Postgres log table.
Key components:
- 4 metric cards: tokens saved, avg latency, stop rate %, cost saved ($)
- Entropy trace chart (Recharts LineChart) — shows H(t) per token + EMA line,
  with a vertical marker at the stop point
- Threshold calibration sliders — one per TaskType, live-updating Redis
- Query log table — last 50 requests with task type, tokens generated, early_stop bool

### Priority 3 — Embedding-based classifier
Replace the regex `classify()` with a proper ML router:
- Use `text-embedding-3-small` to embed incoming prompts
- Cluster into task types using k-means or a lightweight linear classifier
- Train on a labeled dataset of ~500 prompts per TaskType
- This is the "week 2" upgrade that makes the project significantly more impressive

### Priority 4 — Entropy profile matching
Instead of a single threshold per task type, store the full entropy curve shape
for each task and use DTW (dynamic time warping) distance to match new queries
to known profiles. More faithful to the EDRM paper's "manifold" framing.

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
