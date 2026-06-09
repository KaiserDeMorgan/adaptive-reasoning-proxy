"use client";

import { useEffect, useState } from "react";
import { Thresholds, setThreshold } from "@/lib/api";

export default function ThresholdSliders({
  thresholds,
}: {
  thresholds: Thresholds | null;
}) {
  // Local mirror so dragging feels instant; pushes to the proxy on change.
  const [local, setLocal] = useState<Thresholds>({});

  useEffect(() => {
    if (thresholds) setLocal((prev) => ({ ...thresholds, ...prev }));
    // Only seed from server values we don't already control locally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [thresholds]);

  const onChange = (task: string, value: number) => {
    setLocal((p) => ({ ...p, [task]: value }));
  };

  const onCommit = (task: string, value: number) => {
    setThreshold(task, value).catch(() => {});
  };

  const tasks = Object.keys(local).length
    ? Object.keys(local)
    : thresholds
    ? Object.keys(thresholds)
    : [];

  return (
    <div className="panel">
      <h2>Threshold Calibration</h2>
      {tasks.length === 0 ? (
        <div className="empty">Loading thresholds…</div>
      ) : (
        <div className="sliders">
          {tasks.map((task) => {
            const v = local[task] ?? thresholds?.[task] ?? 0.5;
            return (
              <div className="slider-row" key={task}>
                <div className="top">
                  <span style={{ textTransform: "capitalize" }}>{task}</span>
                  <span className="tval">{v.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.01}
                  value={v}
                  onChange={(e) => onChange(task, parseFloat(e.target.value))}
                  onMouseUp={(e) =>
                    onCommit(task, parseFloat((e.target as HTMLInputElement).value))
                  }
                  onTouchEnd={(e) =>
                    onCommit(task, parseFloat((e.target as HTMLInputElement).value))
                  }
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
