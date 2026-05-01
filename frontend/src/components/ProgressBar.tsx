type ProgressBarProps = {
  value?: number | null;
};

export default function ProgressBar({ value }: ProgressBarProps) {
  const percentage = value ? Math.min(100, Math.max(0, value * 100)) : 0;
  return (
    <div className="progress">
      <span style={{ width: `${percentage}%` }} />
    </div>
  );
}
