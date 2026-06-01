import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from entropy_engine import EntropyState
from classifier import classify, THRESHOLDS
from threshold_store import get_threshold

app = FastAPI(title="Adaptive Reasoning Proxy")
client = AsyncOpenAI()  # reads OPENAI_API_KEY from environment


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
    task = classify(prompt)
    threshold = await get_threshold(task.value, THRESHOLDS[task])

    # Force logprobs on — client doesn't need to ask
    body["logprobs"] = True
    body["top_logprobs"] = 5
    body["stream"] = True

    async def generate():
        state = EntropyState()
        t0 = time.monotonic()

        stream = await client.chat.completions.create(**body)
        async for chunk in stream:
            delta = chunk.choices[0].delta
            logprobs = chunk.choices[0].logprobs

            if logprobs and logprobs.content:
                for lp_obj in logprobs.content:
                    token_lps = {t.token: t.logprob for t in lp_obj.top_logprobs}
                    state.update(token_lps)

                if state.should_stop(threshold):
                    state.stopped_early = True
                    meta = {
                        "early_stop": True,
                        "task_type": task.value,
                        "final_entropy_ema": round(state.ema, 4),
                        "tokens_generated": state.tokens_generated,
                        "latency_ms": round((time.monotonic() - t0) * 1000),
                    }
                    stop_chunk = {
                        "choices": [{"delta": {}, "finish_reason": "entropy_stop"}],
                        "edrm_meta": meta,
                    }
                    yield f"data: {json.dumps(stop_chunk)}\n\ndata: [DONE]\n\n"
                    return

            yield f"data: {chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
