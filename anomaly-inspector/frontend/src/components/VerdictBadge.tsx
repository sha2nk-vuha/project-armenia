import { CheckCircle2, XCircle } from "lucide-react";
import { cn } from "../lib/utils";

interface Props {
  verdict: "ok" | "not_ok" | null;
  score: number | null;
  scoreLabel?: string;
}

export function VerdictBadge({ verdict, score, scoreLabel = "Score" }: Props) {
  if (verdict === null) {
    return (
      <div className="flex items-center justify-center h-20 rounded-2xl bg-gray-50 border border-dashed border-gray-200">
        <p className="text-sm text-gray-400">Run inference to see result</p>
      </div>
    );
  }

  const isOk = verdict === "ok";
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 h-20 rounded-2xl border-2",
        isOk
          ? "bg-green-50 border-green-300 text-green-700"
          : "bg-red-50 border-red-300 text-red-700"
      )}
    >
      <div className="flex items-center gap-2">
        {isOk ? (
          <CheckCircle2 className="h-6 w-6" />
        ) : (
          <XCircle className="h-6 w-6" />
        )}
        <span className="text-xl font-bold">{isOk ? "OK" : "NOT OK"}</span>
      </div>
      {score !== null && (
        <span className="text-xs font-mono opacity-80">
          {scoreLabel}: {score.toFixed(4)}
        </span>
      )}
    </div>
  );
}
