import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from openai import AsyncOpenAI

import query_log
from classifier import THRESHOLDS, TaskType
from embedding_classifier import classify_async
from entropy_engine import EntropyState
from entropy_profiles import classify_by_profile, profile_summary, record_profile
from threshold_store import (
    all_thresholds,
    get_threshold,
    record_outcome,
    set_threshold,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# String-keyed default thresholds (task value -> threshold) for the API layer.
DEFAULT_THRESHOLDS: dict[str, float] = {t.value: v for t, v in THRESHOLDS.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await query_log.init_db()
    yield


app = FastAPI(title="Adaptive Reasoning Proxy", lifespan=lifespan)

# The Next.js dashboard runs on a different origin (localhost:3000) in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
    api_key=os.getenv("LLM_API_KEY", "none"),
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def proxy(request: Request):
    """
    Drop-in replacement for the OpenAI chat completions endpoint.
    Clients change one base URL — nothing else.
    """
    body = await request.json()
    prompt = body["messages"][-1]["content"]
    task = await classify_async(prompt)
    threshold = await get_threshold(task.value, THRESHOLDS[task])

    # Force logprobs on so the entropy engine has signal
    body["logprobs"] = True
    body["top_logprobs"] = 5
    body["stream"] = True

    async def generate():
        state = EntropyState()
        trace: list[float] = []
        ema_trace: list[float] = []
        t0 = time.monotonic()

        stream = await client.chat.completions.create(**body)
        async for chunk in stream:
            delta = chunk.choices[0].delta
            logprobs = chunk.choices[0].logprobs

            if logprobs and logprobs.content:
                for lp_obj in logprobs.content:
                    token_lps = {t.token: t.logprob for t in lp_obj.top_logprobs}
                    h = state.update(token_lps)
                    trace.append(round(h, 4))
                    ema_trace.append(round(state.ema, 4))

                if state.should_stop(threshold):
                    state.stopped_early = True
                    latency_ms = round((time.monotonic() - t0) * 1000)
                    meta = {
                        "early_stop": True,
                        "task_type": task.value,
                        "final_entropy_ema": round(state.ema, 4),
                        "tokens_generated": state.tokens_generated,
                        "latency_ms": latency_ms,
                    }
                    await _finalize(
                        task.value, state, trace, ema_trace,
                        stop_index=len(trace), latency_ms=latency_ms, threshold=threshold,
                    )
                    stop_chunk = {
                        "choices": [{"delta": {}, "finish_reason": "entropy_stop"}],
                        "edrm_meta": meta,
                    }
                    yield f"data: {json.dumps(stop_chunk)}\n\ndata: [DONE]\n\n"
                    return

            yield f"data: {chunk.model_dump_json()}\n\n"

        # Stream ended naturally (model emitted its own stop).
        latency_ms = round((time.monotonic() - t0) * 1000)
        await _finalize(
            task.value, state, trace, ema_trace,
            stop_index=None, latency_ms=latency_ms, threshold=threshold,
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _finalize(task_value, state, trace, ema_trace, stop_index, latency_ms, threshold):
    """Persist metrics, the live trace, and the learned entropy profile."""
    query_log.set_latest_trace(task_value, trace, ema_trace, stop_index, threshold)
    await query_log.log_query(
        task_type=task_value,
        tokens_generated=state.tokens_generated,
        early_stop=state.stopped_early,
        latency_ms=latency_ms,
        final_entropy_ema=round(state.ema, 4),
    )
    if trace:
        record_profile(task_value, trace)


# --- dashboard data API -----------------------------------------------------


@app.get("/stats")
async def stats():
    """Aggregate metrics for the dashboard metric cards."""
    return await query_log.stats()


@app.get("/logs")
async def logs(limit: int = 50):
    """Most recent requests for the query log table."""
    return await query_log.recent(limit)


@app.get("/trace/latest")
async def latest_trace():
    """The most recent request's per-token entropy trace for the chart."""
    return query_log.get_latest_trace()


@app.get("/thresholds")
async def thresholds():
    """Current per-task thresholds for the calibration sliders."""
    return await all_thresholds(DEFAULT_THRESHOLDS)


@app.post("/thresholds/{task}")
async def update_threshold(task: str, payload: dict):
    """Set a task's threshold (slider drag). Body: {"value": 0.42}."""
    if task not in DEFAULT_THRESHOLDS:
        raise HTTPException(status_code=404, detail=f"unknown task '{task}'")
    if "value" not in payload:
        raise HTTPException(status_code=422, detail="missing 'value'")
    new_val = await set_threshold(task, payload["value"])
    return {"task": task, "threshold": new_val}


@app.post("/feedback")
async def feedback(payload: dict):
    """Record a user rating to nudge a threshold via the feedback loop.

    Body: {"task": "factual", "entropy_at_stop": 0.3, "rating": 1}.
    """
    task = payload.get("task")
    if task not in DEFAULT_THRESHOLDS:
        raise HTTPException(status_code=404, detail=f"unknown task '{task}'")
    new_val = await record_outcome(
        task, float(payload.get("entropy_at_stop", 0.0)), int(payload.get("rating", 0))
    )
    return {"task": task, "threshold": new_val}


@app.get("/profiles")
async def profiles():
    """Summary of learned entropy profiles per task (DTW reference curves)."""
    return profile_summary()
