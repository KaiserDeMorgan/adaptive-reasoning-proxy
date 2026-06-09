import { LogRow } from "@/lib/api";

function fmtTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

export default function QueryLogTable({ logs }: { logs: LogRow[] }) {
  return (
    <div className="panel">
      <h2>Query Log</h2>
      {logs.length === 0 ? (
        <div className="empty">No requests logged yet.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Task</th>
              <th>Tokens</th>
              <th>Outcome</th>
              <th>Latency</th>
              <th>Final EMA</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((r, i) => (
              <tr key={i}>
                <td>{fmtTime(r.ts)}</td>
                <td>
                  <span className="tag">{r.task_type}</span>
                </td>
                <td>{r.tokens_generated}</td>
                <td>
                  {r.early_stop ? (
                    <span className="badge-stop">early stop</span>
                  ) : (
                    <span className="badge-full">full</span>
                  )}
                </td>
                <td>{r.latency_ms} ms</td>
                <td>{r.final_entropy_ema.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
