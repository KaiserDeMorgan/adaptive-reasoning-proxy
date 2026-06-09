"use client";

import { useCallback, useEffect, useState } from "react";
import MetricCards from "@/components/MetricCards";
import EntropyChart from "@/components/EntropyChart";
import ThresholdSliders from "@/components/ThresholdSliders";
import QueryLogTable from "@/components/QueryLogTable";
import {
  LogRow,
  Stats,
  Thresholds,
  Trace,
  getLogs,
  getStats,
  getThresholds,
  getTrace,
} from "@/lib/api";

const POLL_MS = 2000;

export default function Home() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [online, setOnline] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [s, t, l, th] = await Promise.all([
        getStats(),
        getTrace(),
        getLogs(50),
        getThresholds(),
      ]);
      setStats(s);
      setTrace(t);
      setLogs(l);
      setThresholds(th);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <main className="container">
      <div className="header">
        <div>
          <h1>Adaptive Reasoning Proxy</h1>
          <div className="sub">Entropy-phase early stopping · live metrics</div>
        </div>
        <div className="sub">
          <span className="dot" style={{ background: online ? "#4fd18b" : "#c0506a" }} />
          {online ? "connected" : "proxy offline"}
        </div>
      </div>

      <MetricCards stats={stats} />
      <EntropyChart trace={trace} />
      <ThresholdSliders thresholds={thresholds} />
      <QueryLogTable logs={logs} />
    </main>
  );
}
