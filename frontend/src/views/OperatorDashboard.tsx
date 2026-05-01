import { useEffect, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { apiFetch } from "../api/client";
import StatCard from "../components/StatCard";
import StatusBadge from "../components/StatusBadge";
import { useLiveStream } from "../hooks/useLiveStream";

type OperatorOverview = {
  kpis: Record<string, number | null>;
  grid_health: Array<Record<string, any>>;
  duck_curve: Array<Record<string, any>>;
  risk_timeline: Array<Record<string, any>>;
  signal_history: Array<Record<string, any>>;
};

export default function OperatorDashboard() {
  const [overview, setOverview] = useState<OperatorOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [signalType, setSignalType] = useState("THROTTLE");
  const [severity, setSeverity] = useState(0.5);
  const [targets, setTargets] = useState("1,2,3");
  const [reason, setReason] = useState("Manual override");
  const [signalResponse, setSignalResponse] = useState<string | null>(null);
  const live = useLiveStream();

  useEffect(() => {
    apiFetch<OperatorOverview>("/api/dashboard/operator/overview")
      .then(setOverview)
      .catch((err) => setError((err as Error).message));
  }, []);

  const sendSignal = async () => {
    setSignalResponse(null);
    try {
      const targetNodeIds = targets
        .split(",")
        .map((value) => Number(value.trim()))
        .filter((value) => !Number.isNaN(value));
      const response = await apiFetch<{ signal_id: string }>("/api/signals", {
        method: "POST",
        body: JSON.stringify({
          signal_type: signalType,
          severity,
          target_node_ids: targetNodeIds,
          reason,
        }),
      });
      setSignalResponse(`Signal queued: ${response.signal_id}`);
    } catch (err) {
      setSignalResponse((err as Error).message);
    }
  };

  return (
    <div className="container">
      <div className="section-title">
        <h2>Operator Overview</h2>
        <div className="muted">Live stream: {live.status}</div>
      </div>

      {error && <div className="warning">{error}</div>}

      {overview && (
        <div className="grid grid-3">
          <StatCard label="Curtailment" value={`${(overview.kpis.curtailment_pct ?? 0) * 100}%`} />
          <StatCard
            label="Peak Reduction"
            value={`${(overview.kpis.peak_reduction_pct ?? 0) * 100}%`}
          />
          <StatCard
            label="Transformer Aging Rate"
            value={overview.kpis.transformer_aging_rate ?? "N/A"}
          />
          <StatCard
            label="P2P Volume (kWh)"
            value={Number(overview.kpis.p2p_volume_kwh ?? 0).toFixed(2)}
          />
          <StatCard
            label="P2P Value (₹)"
            value={Number(overview.kpis.p2p_value_inr ?? 0).toFixed(2)}
          />
          <StatCard label="Overload Events" value={overview.kpis.overload_events ?? 0} />
        </div>
      )}

      <div className="grid grid-2" style={{ marginTop: "24px" }}>
        <div className="card">
          <h3>Grid Health Map</h3>
          <div className="grid" style={{ gap: "12px" }}>
            {overview?.grid_health?.map((node) => (
              <div key={node.node_id} className="card">
                <div className="flex" style={{ justifyContent: "space-between" }}>
                  <div>
                    <div style={{ fontWeight: 600 }}>Node {node.node_id}</div>
                    <div className="muted">{node.node_type}</div>
                  </div>
                  <StatusBadge level={node.stress_level || "low"} />
                </div>
                <div className="muted">Voltage: {node.voltage_pu ?? "--"}</div>
                <div className="muted">Load: {node.household_load_kw ?? "--"} kW</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Manual Override Panel</h3>
          <div className="grid" style={{ gap: "12px" }}>
            <label>
              <div className="muted">Signal type</div>
              <input
                className="input"
                value={signalType}
                onChange={(event) => setSignalType(event.target.value)}
              />
            </label>
            <label>
              <div className="muted">Severity (0-1)</div>
              <input
                className="input"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={severity}
                onChange={(event) => setSeverity(Number(event.target.value))}
              />
            </label>
            <label>
              <div className="muted">Target nodes</div>
              <input
                className="input"
                value={targets}
                onChange={(event) => setTargets(event.target.value)}
              />
            </label>
            <label>
              <div className="muted">Reason</div>
              <input
                className="input"
                value={reason}
                onChange={(event) => setReason(event.target.value)}
              />
            </label>
            <button className="button" onClick={sendSignal}>
              Broadcast Signal
            </button>
            {signalResponse && <div className="muted">{signalResponse}</div>}
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: "24px" }}>
        <div className="card">
          <h3>Duck Curve Tracker</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={overview?.duck_curve || []}>
              <XAxis dataKey="ts" hide />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="actual_kw" stroke="#38bdf8" />
              <Line type="monotone" dataKey="forecast_kw" stroke="#a855f7" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Risk Timeline</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={overview?.risk_timeline || []}>
              <XAxis dataKey="ts" hide />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="overload_probability" stroke="#f87171" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginTop: "24px" }}>
        <h3>Signal History</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Type</th>
              <th>Severity</th>
              <th>Targets</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {overview?.signal_history?.map((signal) => (
              <tr key={signal.signal_id}>
                <td>{signal.ts}</td>
                <td>{signal.signal_type}</td>
                <td>{signal.severity}</td>
                <td>{JSON.stringify(signal.target_node_ids)}</td>
                <td>{signal.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
