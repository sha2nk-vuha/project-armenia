import { Label } from "./ui/label";
import { ParamField, selectClass } from "./RuleParamFields";
import type {
  CalibrationState,
  DecisionRuleInfo,
  RuleParams,
} from "../api/client";
import { Button } from "./ui/button";

interface Props {
  rules: DecisionRuleInfo[];
  labels: Record<string, string>;
  selected: string | null;
  params: RuleParams;
  disabled?: boolean;
  // Per-SKU calibration, for rules that declare a baseline.
  calibration: CalibrationState | null;
  calibrationGroups: string[];
  calibrationGroup: string | null;
  calibrating: boolean;
  canCalibrate: boolean;
  onSelect: (rule: string) => void;
  onParamChange: (name: string, value: unknown) => void;
  onCalibrationGroupChange: (group: string) => void;
  onCalibrate: () => void;
  onClearCalibration: () => void;
}

export function DecisionRuleControl({
  rules,
  labels,
  selected,
  params,
  disabled,
  calibration,
  calibrationGroups,
  calibrationGroup,
  calibrating,
  canCalibrate,
  onSelect,
  onParamChange,
  onCalibrationGroupChange,
  onCalibrate,
  onClearCalibration,
}: Props) {
  if (rules.length === 0) return null;
  const active = rules.find((r) => r.name === selected) ?? null;
  const spec = active?.calibration ?? null;

  return (
    <div className="space-y-4 border-t border-gray-100 pt-4">
      <div className="space-y-1.5">
        <Label htmlFor="decision-rule">Decision Rule</Label>
        <select
          id="decision-rule"
          value={selected ?? ""}
          disabled={disabled}
          onChange={(e) => onSelect(e.target.value)}
          className={selectClass}
        >
          {rules.map((r) => (
            <option key={r.name} value={r.name}>
              {r.label}
            </option>
          ))}
        </select>
        <p className="text-[11px] text-gray-400">
          How the model's output becomes an OK / NOT OK verdict.
        </p>
      </div>

      {active?.params.map((p) => (
        <div key={p.name} className="space-y-1">
          <ParamField
            spec={p}
            value={params[p.name]}
            labels={labels}
            disabled={disabled}
            onChange={(v) => onParamChange(p.name, v)}
          />
          {p.help && <p className="text-[11px] text-gray-400">{p.help}</p>}
        </div>
      ))}

      {spec && (
        <div className="space-y-2 rounded-lg border border-gray-200 bg-gray-50 p-3">
          <div className="flex items-center justify-between">
            <Label>Calibration</Label>
            <span
              className={`text-[11px] font-medium ${
                calibration?.calibrated ? "text-green-600" : "text-gray-400"
              }`}
            >
              {calibration?.calibrated ? "Taught" : "Not calibrated"}
            </span>
          </div>

          {calibration?.calibrated ? (
            <p className="text-[11px] text-gray-500 leading-snug">
              {spec.label} {String(calibration.params[spec.param] ?? "—")} from{" "}
              {calibration.sample_count} sample
              {calibration.sample_count === 1 ? "" : "s"}
              {typeof calibration.spread === "number" && (
                <>
                  , spread {calibration.spread.toFixed(4)}
                  {/* A tolerance at or below the spread of known-good samples is
                      measuring noise, so say so rather than let it look fine. */}
                  {typeof params.max_offset_ratio === "number" &&
                    calibration.spread >= (params.max_offset_ratio as number) && (
                      <span className="block text-amber-600">
                        Spread exceeds the tolerance — the good samples vary more
                        than the limit allows.
                      </span>
                    )}
                </>
              )}
            </p>
          ) : (
            <p className="text-[11px] text-gray-400 leading-snug">
              Teach this SKU's {spec.label.toLowerCase()} from images known to be
              good, so one tolerance can serve every SKU.
            </p>
          )}

          <select
            value={calibrationGroup ?? ""}
            disabled={disabled || calibrationGroups.length === 0}
            onChange={(e) => onCalibrationGroupChange(e.target.value)}
            className={selectClass}
          >
            {calibrationGroups.length === 0 && <option value="">No image groups</option>}
            {calibrationGroups.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>

          <div className="flex gap-2">
            <Button
              size="sm"
              className="flex-1"
              disabled={!canCalibrate || calibrating}
              onClick={onCalibrate}
            >
              {calibrating ? "Teaching…" : "Teach from group"}
            </Button>
            {calibration?.calibrated && (
              <Button size="sm" variant="ghost" onClick={onClearCalibration}>
                Clear
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
