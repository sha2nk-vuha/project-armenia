import type { StatsResponse } from "../api/client";

interface StatCardProps {
  label: string;
  value: string | number;
  accent?: string;
}

function StatCard({ label, value, accent = "text-gray-800" }: StatCardProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg bg-gray-50 border border-gray-200 px-4 py-3 flex-1">
      <span className={`text-2xl font-bold ${accent}`}>{value}</span>
      <span className="text-xs text-gray-500 mt-0.5">{label}</span>
    </div>
  );
}

interface Props {
  stats: StatsResponse | null;
}

export function StatsPanel({ stats }: Props) {
  const total = stats?.total ?? 0;
  const ok = stats?.ok ?? 0;
  const notOk = stats?.not_ok ?? 0;
  const passRate = stats?.pass_rate ?? 0;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
        Statistics
      </h3>
      <div className="flex gap-2">
        <StatCard label="Total" value={total} />
        <StatCard label="OK" value={ok} accent="text-green-600" />
        <StatCard label="NOT OK" value={notOk} accent="text-red-600" />
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 rounded-full bg-gray-200 overflow-hidden">
          <div
            className="h-full bg-green-500 rounded-full transition-all duration-500"
            style={{ width: `${passRate}%` }}
          />
        </div>
        <span className="text-sm font-medium text-gray-600 min-w-[52px] text-right">
          {passRate.toFixed(1)}% pass
        </span>
      </div>
    </div>
  );
}
