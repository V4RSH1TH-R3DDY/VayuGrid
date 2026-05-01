import React from "react";

type StatCardProps = {
  label: string;
  value: React.ReactNode;
  footnote?: string;
};

export default function StatCard({ label, value, footnote }: StatCardProps) {
  return (
    <div className="card">
      <div className="muted">{label}</div>
      <div style={{ fontSize: "24px", fontWeight: 700, marginTop: "6px" }}>{value}</div>
      {footnote && <div className="muted">{footnote}</div>}
    </div>
  );
}
