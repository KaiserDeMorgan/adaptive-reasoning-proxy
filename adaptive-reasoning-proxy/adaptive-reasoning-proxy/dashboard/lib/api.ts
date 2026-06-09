// Thin client for the Adaptive Reasoning Proxy API.
// Override the proxy location with NEXT_PUBLIC_PROXY_URL at build/run time.

export const PROXY_URL =
  process.env.NEXT_PUBLIC_PROXY_URL || "http://localhost:8000";

export interface Stats {
  total_queries: number;
  early_stops: number;
  stop_rate: number;
  avg_latency_ms: number;
  tokens_saved: number;
  cost_saved_usd: number;
  baseline_tokens: number;
}

export interface Trace {
  task_type: string | null;
  entropy: number[];
  ema: number[];
  stop_index: number | null;
  threshold: number | null;
}

export interface LogRow {
  ts: number;
  task_type: string;
  tokens_generated: number;
  early_stop: boolean;
  latency_ms: number;
  final_entropy_ema: number;
}

export type Thresholds = Record<string, number>;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${PROXY_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const getStats = () => getJSON<Stats>("/stats");
export const getTrace = () => getJSON<Trace>("/trace/latest");
export const getLogs = (limit = 50) => getJSON<LogRow[]>(`/logs?limit=${limit}`);
export const getThresholds = () => getJSON<Thresholds>("/thresholds");

export async function setThreshold(task: string, value: number): Promise<void> {
  await fetch(`${PROXY_URL}/thresholds/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
}
