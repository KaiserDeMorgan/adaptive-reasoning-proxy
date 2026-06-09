import { Stats } from "@/lib/api";

function Card({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

export default function MetricCards({ stats }: { stats: Stats | null }) {
  const s = stats;
  return (
    <div className="grid-cards">
      <Card
        label="Tokens Saved"
        value={s ? s.tokens_saved.toLocaleString() : "—"}
        hint={s ? `vs ${s.baseline_tokens}-token baseline` : undefined}
      />
      <Card
        label="Avg Latency"
        value={s ? `${s.avg_latency_ms} ms` : "—"}
        hint="per request"
      />
      <Card
        label="Stop Rate"
        value={s ? `${(s.stop_rate * 100).toFixed(0)}%` : "—"}
        hint={s ? `${s.early_stops}/${s.total_queries} requests` : undefined}
      />
      <Card
        label="Cost Saved"
        value={s ? `$${s.cost_saved_usd.toFixed(2)}` : "—"}
        hint="estimated"
      />
    </div>
  );
}
