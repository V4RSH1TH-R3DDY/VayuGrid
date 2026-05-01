type StatusBadgeProps = {
  level: "low" | "medium" | "high" | "critical" | string;
};

export default function StatusBadge({ level }: StatusBadgeProps) {
  return <span className={`badge ${level}`}>{level}</span>;
}
