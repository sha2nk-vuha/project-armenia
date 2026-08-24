import { CheckCircle2, MinusCircle, XCircle } from "lucide-react";
import type { StageResult } from "../api/client";

/** The ordered per-stage breakdown of a cascade result. */
export function CascadeStages({ stages }: { stages: StageResult[] }) {
  if (stages.length === 0) return null;
  return (
    <div className="rounded-xl border border-gray-200 overflow-hidden">
      <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wider">
        Stages ({stages.length})
      </div>
      <ul className="divide-y divide-gray-100">
        {stages.map((stage, i) => (
          <li key={i} className={`px-3 py-2 ${stage.evaluated ? "" : "opacity-60"}`}>
            <div className="flex items-center gap-2">
              <StageIcon verdict={stage.verdict} />
              <span className="text-sm font-medium text-gray-700">
                {i + 1}. {stage.feature.replace(/_/g, " ")}
              </span>
              <span className="text-[11px] text-gray-400">/ {stage.decision_rule.replace(/_/g, " ")}</span>
              {stage.score !== null && (
                <span className="ml-auto font-mono text-xs text-gray-500">
                  {stage.score_label}: {stage.score.toFixed(4)}
                </span>
              )}
            </div>
            <p className="mt-0.5 pl-6 text-[11px] text-gray-500 leading-snug">{stage.reason}</p>
            {stage.image && (
              <img
                src={`data:image/jpeg;base64,${stage.image}`}
                alt={`stage ${i + 1}`}
                className="mt-2 ml-6 max-h-40 rounded-lg border border-gray-200 object-contain"
              />
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function StageIcon({ verdict }: { verdict: StageResult["verdict"] }) {
  if (verdict === "ok") return <CheckCircle2 className="h-4 w-4 text-green-600" />;
  if (verdict === "not_ok") return <XCircle className="h-4 w-4 text-red-600" />;
  return <MinusCircle className="h-4 w-4 text-gray-400" />; // skipped
}
