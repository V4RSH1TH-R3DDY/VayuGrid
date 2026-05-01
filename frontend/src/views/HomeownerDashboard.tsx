import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiFetch, authStore } from "../api/client";
import ProgressBar from "../components/ProgressBar";
import StatCard from "../components/StatCard";
import { useLiveStream } from "../hooks/useLiveStream";

type HomeownerSummary = {
  node_id: number;
  today_energy: Record<string, number>;
  ev_status: Record<string, any>;
  battery_health: Record<string, any>;
  earnings: Record<string, number>;
  live_market: Array<Record<string, any>>;
};

type ConsentState = {
  node_id: number;
  consented: boolean;
  consent_version: string;
  categories: string[];
};

const CONSENT_OPTIONS = ["telemetry", "market", "device", "billing"];

export default function HomeownerDashboard() {
  const params = useParams();
  const storedNodeId = authStore.getNodeId();
  const nodeId = Number(params.nodeId || storedNodeId || 1);
  const [summary, setSummary] = useState<HomeownerSummary | null>(null);
  const [consent, setConsent] = useState<ConsentState | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useLiveStream();

  useEffect(() => {
    setError(null);
    apiFetch<HomeownerSummary>(`/api/dashboard/homeowner/${nodeId}/summary`)
      .then(setSummary)
      .catch((err) => setError((err as Error).message));
    apiFetch<ConsentState>(`/api/privacy/consent/${nodeId}`)
      .then(setConsent)
      .catch((err) => setError((err as Error).message));
  }, [nodeId]);

  const consented = consent?.consented ?? false;

  const toggleCategory = (category: string) => {
    if (!consent) return;
    const next = consent.categories.includes(category)
      ? consent.categories.filter((value) => value !== category)
      : [...consent.categories, category];
    setConsent({ ...consent, categories: next });
  };

  const saveConsent = async () => {
    if (!consent) return;
    await apiFetch(`/api/privacy/consent/${nodeId}`, {
      method: "POST",
      body: JSON.stringify({
        consented: consent.consented,
        consent_version: consent.consent_version || "v1",
        categories: consent.categories,
      }),
    });
    setMessage("Consent saved.");
  };

  const requestExport = async () => {
    const data = await apiFetch(`/api/privacy/export/${nodeId}`);
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `vayugrid_node_${nodeId}_export.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const requestDeletion = async () => {
    const response = await apiFetch<{
      request_id: string;
      scheduled_for: string;
    }>(`/api/privacy/delete/${nodeId}`, { method: "POST" });
    setMessage(`Deletion scheduled for ${response.scheduled_for}`);
  };

  const evProgress = useMemo(
    () => summary?.ev_status?.progress_pct ?? 0,
    [summary],
  );

  return (
    <div className="container">
      <div className="section-title">
        <h2>Homeowner View</h2>
        <div className="muted">Live stream: {live.status}</div>
      </div>

      {error && <div className="warning">{error}</div>}

      {summary && (
        <div className="grid grid-3">
          <StatCard
            label="Solar Generated (kWh)"
            value={summary.today_energy.solar_generated_kwh.toFixed(2)}
          />
          <StatCard
            label="Home Consumed (kWh)"
            value={summary.today_energy.home_consumed_kwh.toFixed(2)}
          />
          <StatCard
            label="Grid Imported (kWh)"
            value={summary.today_energy.grid_imported_kwh.toFixed(2)}
          />
          <StatCard
            label="P2P Sold (kWh)"
            value={summary.today_energy.p2p_sold_kwh.toFixed(2)}
          />
          <StatCard
            label="P2P Bought (kWh)"
            value={summary.today_energy.p2p_bought_kwh.toFixed(2)}
          />
          <StatCard
            label="Net Bill Impact (₹)"
            value={summary.today_energy.net_bill_inr.toFixed(2)}
          />
        </div>
      )}

      <div className="grid grid-2" style={{ marginTop: "24px" }}>
        <div className="card">
          <h3>EV Status</h3>
          <div className="muted">
            Current charge: {summary?.ev_status?.current_kwh ?? "--"} kWh
          </div>
          <div className="muted">
            Target: {summary?.ev_status?.target_kwh ?? "--"} kWh
          </div>
          <div className="muted">
            Deadline: {summary?.ev_status?.deadline ?? "--"}
          </div>
          <ProgressBar value={evProgress} />
        </div>
        <div className="card">
          <h3>Battery Health</h3>
          <div className="muted">
            SoC: {summary?.battery_health?.soc_kwh ?? "--"} kWh
          </div>
          <div className="muted">
            Capacity: {summary?.battery_health?.capacity_kwh ?? "--"} kWh
          </div>
          <div className="muted">
            Health: {summary?.battery_health?.health_pct ?? "--"}%
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: "24px" }}>
        <div className="card">
          <h3>Earnings</h3>
          <div className="muted">
            P2P revenue: ₹{summary?.earnings?.p2p_revenue_inr}
          </div>
          <div className="muted">
            P2P cost: ₹{summary?.earnings?.p2p_cost_inr}
          </div>
          <div className="muted">
            Net-metering baseline: ₹
            {summary?.earnings?.net_metering_revenue_inr}
          </div>
          <div className="muted">
            Delta vs net-metering: ₹
            {summary?.earnings?.delta_vs_net_metering_inr}
          </div>
        </div>
        <div className="card">
          <h3>Live Market</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={summary?.live_market || []}>
              <XAxis dataKey="ts" hide />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="avg_price" stroke="#22d3ee" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: "24px" }}>
        <div className="card">
          <h3>Consent & Privacy</h3>
          <label className="flex">
            <input
              type="checkbox"
              checked={consented}
              onChange={(event) =>
                setConsent((current) =>
                  current
                    ? { ...current, consented: event.target.checked }
                    : current,
                )
              }
            />
            <span>Allow VayuGrid to process my data for optimization.</span>
          </label>
          <div className="muted" style={{ marginTop: "12px" }}>
            Data categories you agree to share:
          </div>
          <div className="flex" style={{ marginTop: "8px" }}>
            {CONSENT_OPTIONS.map((option) => (
              <label key={option} className="flex">
                <input
                  type="checkbox"
                  checked={consent?.categories?.includes(option) ?? false}
                  onChange={() => toggleCategory(option)}
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
          <div className="flex" style={{ marginTop: "12px" }}>
            <button className="button" onClick={saveConsent}>
              Save Consent
            </button>
            <button className="button secondary" onClick={requestExport}>
              Download Data
            </button>
            <button className="button danger" onClick={requestDeletion}>
              Request Deletion
            </button>
          </div>
          {message && <div className="muted">{message}</div>}
        </div>
        <div className="card">
          <h3>Today's Energy Flow</h3>
          <div className="muted">
            Solar → Home: {summary?.today_energy?.home_consumed_kwh ?? 0} kWh
          </div>
          <div className="muted">
            Solar → Neighbors: {summary?.today_energy?.p2p_sold_kwh ?? 0} kWh
          </div>
          <div className="muted">
            Grid → Home: {summary?.today_energy?.grid_imported_kwh ?? 0} kWh
          </div>
          <div className="muted">
            Neighbors → Home: {summary?.today_energy?.p2p_bought_kwh ?? 0} kWh
          </div>
        </div>
      </div>
    </div>
  );
}
