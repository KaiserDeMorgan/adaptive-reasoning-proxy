# Adaptive Reasoning Proxy — Dashboard

Real-time Next.js + Recharts dashboard for the proxy. Polls the proxy's data
endpoints every 2s and renders:

- **Metric cards** — tokens saved, avg latency, stop rate %, cost saved
- **Entropy trace chart** — H(t) per token + EMA line, with markers for the
  threshold and the early-stop point
- **Threshold calibration sliders** — one per task type, writing live to the proxy
- **Query log table** — the last 50 requests

## Prerequisites

[Node.js](https://nodejs.org/) 18+ (LTS recommended). The proxy must be running
on `http://localhost:8000` (see the project root README).

## Run locally

```bash
cd dashboard
cp .env.local.example .env.local   # optional: change the proxy URL
npm install
npm run dev
```

Open <http://localhost:3000>.

## Configuration

| Env var                 | Default                 | Purpose                       |
| ----------------------- | ----------------------- | ----------------------------- |
| `NEXT_PUBLIC_PROXY_URL` | `http://localhost:8000` | Where to read proxy data from |

## Data sources

| Component         | Endpoint           |
| ----------------- | ------------------ |
| Metric cards      | `GET /stats`       |
| Entropy chart     | `GET /trace/latest`|
| Threshold sliders | `GET/POST /thresholds` |
| Query log table   | `GET /logs`        |

## Deploy

Builds as a standard Next.js app — deploy to Vercel and set
`NEXT_PUBLIC_PROXY_URL` to your deployed proxy's public URL.
