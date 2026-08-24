import * as SliderPrimitive from "@radix-ui/react-slider";
import { Label } from "./ui/label";
import type {
  CalibrationState,
  DecisionRuleInfo,
  ParamSpec,
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

const selectClass =
  "flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm " +
  "focus:outline-none focus:ring-2 focus:ring-yellow-400 disabled:opacity-50 disabled:cursor-not-allowed";

/**
 * Renders one Decision Rule parameter from its schema. Every control here is
 * driven by `spec.type`, so adding a rule on the backend surfaces its controls
 * with no frontend change — that is the point of the seam.
 */
function ParamField({
  spec,
  value,
  labels,
  disabled,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  labels: Record<string, string>;
  disabled?: boolean;
  onChange: (value: unknown) => void;
}) {
  const classNames = Object.values(labels);

  if (spec.type === "number") {
    const num = typeof value === "number" ? value : Number(spec.default ?? 0);
    const min = spec.min ?? 0;
    const max = spec.max ?? 1;
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>{spec.label}</Label>
          <span className="text-sm font-mono font-medium text-yellow-600">
            {num.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span>{min.toFixed(2)}</span>
          <SliderPrimitive.Root
            className="relative flex flex-1 touch-none select-none items-center"
            min={min}
            max={max}
            step={spec.step ?? 0.01}
            value={[num]}
            disabled={disabled}
            onValueChange={([v]) => onChange(v)}
          >
            <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-gray-200">
              <SliderPrimitive.Range className="absolute h-full bg-yellow-400" />
            </SliderPrimitive.Track>
            <SliderPrimitive.Thumb className="block h-4 w-4 rounded-full border-2 border-yellow-400 bg-white shadow transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-300" />
          </SliderPrimitive.Root>
          <span>{max.toFixed(2)}</span>
        </div>
      </div>
    );
  }

  if (spec.type === "bool") {
    return (
      <label className="flex items-center gap-2 text-sm text-gray-700">
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 rounded border-gray-300 accent-yellow-400"
        />
        {spec.label}
      </label>
    );
  }

  if (spec.type === "enum" || spec.type === "class") {
    // `class` options come from the loaded model's Class Catalog, so an
    // operator picks a real class name rather than guessing a numeric id.
    const options = spec.type === "class" ? classNames : spec.options ?? [];
    return (
      <div className="space-y-1.5">
        <Label htmlFor={`param-${spec.name}`}>{spec.label}</Label>
        <select
          id={`param-${spec.name}`}
          value={String(value ?? "")}
          disabled={disabled || options.length === 0}
          onChange={(e) => onChange(e.target.value)}
          className={selectClass}
        >
          {options.length === 0 && <option value="">No classes available</option>}
          {options.map((o) => (
            <option key={o} value={o}>
              {o.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (spec.type === "class_list") {
    const chosen = Array.isArray(value) ? (value as unknown[]).map(String) : [];
    return (
      <div className="space-y-1.5">
        <Label>{spec.label}</Label>
        <div className="rounded-md border border-gray-300 bg-white p-2 space-y-1 max-h-32 overflow-y-auto">
          {classNames.length === 0 && (
            <p className="text-xs text-gray-400">No classes available</p>
          )}
          {classNames.map((name) => (
            <label key={name} className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={chosen.includes(name)}
                disabled={disabled}
                onChange={(e) =>
                  onChange(
                    e.target.checked
                      ? [...chosen, name]
                      : chosen.filter((c) => c !== name)
                  )
                }
                className="h-4 w-4 rounded border-gray-300 accent-yellow-400"
              />
              {name.replace(/_/g, " ")}
            </label>
          ))}
        </div>
      </div>
    );
  }

  return null;
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
