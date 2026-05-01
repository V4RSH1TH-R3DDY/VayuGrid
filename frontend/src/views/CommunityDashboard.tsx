import { useEffect, useState } from "react";
import { apiFetch } from "../api/client";
import StatCard from "../components/StatCard";
import { useLiveStream } from "../hooks/useLiveStream";

type CommunitySummary = {
  backup_hours: number | null;
  community_savings: {
    today_inr: number;
    month_inr: number;
    total_inr: number;
  };
  fairness_allocation: Array<{ priority_tier: string; count: number }>;
};

export default function CommunityDashboard() {
  const [summary, setSummary] = useState<CommunitySummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nodeId, setNodeId] = useState("1");
  const [priority, setPriority] = useState("medical");
  const [reason, setReason] = useState("Critical medical equipment");
  const [message, setMessage] = useState<string | null>(null);
  const live = useLiveStream();

  useEffect(() => {
    setError(null);
    apiFetch<CommunitySummary>("/api/dashboard/community/summary")
      .then(setSummary)
      .catch((err) => setError((err as Error).message));
  }, []);

  const flagCriticalLoad = async () => {
    const response = await apiFetch<{ flag_id: string }>(
      "/api/community/critical-load",
      {
        method: "POST",
        body: JSON.stringify({
          node_id: Number(nodeId),
          priority_tier: priority,
          reason,
        }),
      },
    );
    setMessage(`Flag recorded: ${response.flag_id}`);
  };

  return (
    <div className="container">
      <div className="section-title">
        <h2>Community Resilience</h2>
        <div className="muted">Live stream: {live.status}</div>
      </div>

      {error && <div className="warning">{error}</div>}

      {summary && (
        <div className="grid grid-3">
          <StatCard
            label="Backup Hours Available"
            value={
              summary.backup_hours !== null
                ? `${summary.backup_hours} hrs`
                : "--"
            }
          />
          <StatCard
            label="Savings Today"
            value={`₹${summary.community_savings.today_inr}`}
          />
          <StatCard
            label="Savings This Month"
            value={`₹${summary.community_savings.month_inr}`}
          />
        </div>
      )}

      <div className="grid grid-2" style={{ marginTop: "24px" }}>
        <div className="card">
          <h3>Fairness Allocation</h3>
          {summary?.fairness_allocation?.map((item) => (
            <div key={item.priority_tier} className="muted">
              {item.priority_tier}: {item.count} households
            </div>
          ))}
        </div>
        <div className="card">
          <h3>Flag Critical Load</h3>
          <div className="grid" style={{ gap: "12px" }}>
            <label>
              <div className="muted">Household Node ID</div>
              <input
                className="input"
                value={nodeId}
                onChange={(event) => setNodeId(event.target.value)}
              />
            </label>
            <label>
              <div className="muted">Priority tier</div>
              <input
                className="input"
                value={priority}
                onChange={(event) => setPriority(event.target.value)}
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
            <button className="button" onClick={flagCriticalLoad}>
              Flag Critical Load
            </button>
            {message && <div className="muted">{message}</div>}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: "24px" }}>
        <h3>Community Savings</h3>
        <div className="muted">
          Total savings since launch: ₹{summary?.community_savings.total_inr}
        </div>
      </div>
    </div>
  );
}
