"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Trace } from "@/lib/api";

export default function EntropyChart({ trace }: { trace: Trace | null }) {
  const hasData = trace && trace.entropy && trace.entropy.length > 0;

  const data = hasData
    ? trace!.entropy.map((h, i) => ({
        i,
        h,
        ema: trace!.ema[i],
      }))
    : [];

  return (
    <div className="panel">
      <h2>
        Entropy Trace{" "}
        {trace?.task_type ? <span className="tag">{trace.task_type}</span> : null}
      </h2>
      {!hasData ? (
        <div className="empty">
          No trace yet — send a request through the proxy to populate this chart.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
            <CartesianGrid stroke="#243044" strokeDasharray="3 3" />
            <XAxis
              dataKey="i"
              stroke="#8a97a8"
              fontSize={12}
              label={{ value: "token", position: "insideBottom", fill: "#8a97a8", dy: 12 }}
            />
            <YAxis stroke="#8a97a8" fontSize={12} domain={[0, "auto"]} />
            <Tooltip
              contentStyle={{
                background: "#141a26",
                border: "1px solid #243044",
                borderRadius: 8,
                color: "#e6edf6",
              }}
            />
            {trace?.threshold != null ? (
              <ReferenceLine
                y={trace.threshold}
                stroke="#f0a24b"
                strokeDasharray="5 5"
                label={{ value: "threshold", fill: "#f0a24b", fontSize: 11, position: "right" }}
              />
            ) : null}
            {trace?.stop_index != null ? (
              <ReferenceLine
                x={trace.stop_index - 1}
                stroke="#4fd18b"
                label={{ value: "stop", fill: "#4fd18b", fontSize: 11, position: "top" }}
              />
            ) : null}
            <Line type="monotone" dataKey="h" stroke="#5e8bff" dot={false} name="H(t)" />
            <Line
              type="monotone"
              dataKey="ema"
              stroke="#e6edf6"
              strokeWidth={2}
              dot={false}
              name="EMA"
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
