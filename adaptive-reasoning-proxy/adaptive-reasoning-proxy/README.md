# Adaptive Reasoning Proxy

Tbis is a drop-in FastAPI proxy for any OpenAI-compatible API. Based off the the research paper "(put it in here)" It intercepts the token
stream, computes Shannon entropy over the top-k logprobs in real time, and stops
generation early once the model enters a low-entropy "settled" regime — cutting
completion tokens with minimal accuracy loss. Based on the EDRM entropy
phase-transition framing.

```
client ──> proxy (:8000) ──> llama-server / any OpenAI-compatible API (:8080)
                │
                ├─ entropy engine (H(t), EMA, should_stop)
                ├─ task router (regex)
                ├─ adaptive thresholds (Redis + in-memory fallback)
                ├─ SQLite query log + stats
                └─ DTW entropy-profile learning
                          │
dashboard (:3000) ◀───────┘  (Next.js + Recharts, polls the data API)
```

## Requirements

- **Python 3.11+** (tested on 3.14)
- An OpenAI-compatible LLM server. Locally: `llama-server` from llama.cpp with a
  `.gguf` model that supports logprobs.
- **Node.js 18+** — only needed for the dashboard.
- Redis is **optional** — without it, thresholds use an in-memory + default fallback.

## 1. Start the LLM backend

```powershell
# from the llama.cpp binaries folder, with your model:
.\llama-server.exe -m C:\path\to\model.gguf --port 8080 --ctx-size 4096
```

## 2. Start the proxy

```powershell
copy .env.example .env          # adjust if needed
pip install -r requirements.txt
cd proxy
python -m uvicorn main:app --reload
```

Proxy is now on <http://localhost:8000>. Quick test:

```powershell
curl -X POST http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"m","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

On an early stop the final SSE chunk carries `edrm_meta` with `early_stop: true`,
the token count, latency, and final entropy EMA.

## 3. Start the dashboard (optional)

```powershell
cd dashboard
npm install
npm run dev
```

Open <http://localhost:3000>.

## Run the tests

```powershell
pip install -r requirements.txt
pytest
```

## Data API (consumed by the dashboard)

| Endpoint                  | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `GET /health`             | status check                                       |
| `POST /v1/chat/completions` | drop-in chat endpoint with entropy early-stop    |
| `GET /stats`              | tokens saved, avg latency, stop rate, cost saved   |
| `GET /stats/entropy`      | avg entropy-at-stop per task, for threshold calibration |
| `GET /logs?limit=N`       | recent requests                                    |
| `GET /trace/latest`       | last request's H(t) + EMA + stop index             |
| `GET /thresholds`         | current per-task thresholds                        |
| `POST /thresholds/{task}` | set a threshold (`{"value": 0.42}`)                |
| `POST /feedback`          | nudge a threshold from a rating                    |
| `GET /profiles`           | learned DTW reference curves per task              |

## Configuration

See `.env.example`. Key vars: `LLM_BASE_URL`, `LLM_API_KEY`, `REDIS_URL`,
`BASELINE_TOKENS`, `COST_PER_1K_TOKENS`.

## Docker

```powershell
copy .env.example .env
docker-compose up
```

Brings up proxy + redis + dashboard. The proxy reaches a host-run `llama-server`
via `host.docker.internal:8080`.

## Notes

- The `tokens_saved` / `cost_saved` figures are **estimates** measured against
  `BASELINE_TOKENS` (the assumed uncapped completion length); tune to your data.
- DTW profile matching is an optional enrichment layer that degrades
  gracefully — the proxy works with it disabled.
