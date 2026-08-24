import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { ParamField, selectClass } from "./RuleParamFields";
import type {
  CascadeFeatureOption,
  CascadeOptions,
  CascadeSpec,
  CascadeStageSpec,
  DecisionRuleInfo,
} from "../api/client";

interface Props {
  options: CascadeOptions;
  spec: CascadeSpec;
  disabled?: boolean;
  onChange: (spec: CascadeSpec) => void;
}

/** Seed a stage's params from the rule's schema defaults with sidecar layered on. */
function seedStageParams(rule: DecisionRuleInfo | undefined): Record<string, unknown> {
  if (!rule) return {};
  return {
    ...Object.fromEntries(rule.params.map((p) => [p.name, p.default])),
    ...(rule.defaults ?? {}),
  };
}

function featureOption(options: CascadeOptions, name: string): CascadeFeatureOption | undefined {
  return options.features.find((f) => f.name === name);
}

function ruleOf(feat: CascadeFeatureOption | undefined, name: string): DecisionRuleInfo | undefined {
  return feat?.rules.find((r) => r.name === name);
}

/** A fresh stage for a feature: its default (or first) rule, seeded params. */
function newStage(feat: CascadeFeatureOption): CascadeStageSpec {
  const rule = feat.rules.find((r) => r.name === feat.default_rule) ?? feat.rules[0];
  return {
    feature: feat.name,
    rule: rule?.name ?? "",
    threshold: 0.5,
    params: seedStageParams(rule),
  };
}

export function CascadeBuilder({ options, spec, disabled, onChange }: Props) {
  const patch = (next: Partial<CascadeSpec>) => onChange({ ...spec, ...next });

  const setStage = (i: number, next: CascadeStageSpec) => {
    const stages = spec.stages.slice();
    stages[i] = next;
    patch({ stages });
  };

  const move = (i: number, delta: number) => {
    const j = i + delta;
    if (j < 0 || j >= spec.stages.length) return;
    const stages = spec.stages.slice();
    [stages[i], stages[j]] = [stages[j], stages[i]];
    patch({ stages });
  };

  const addStage = () => {
    if (options.features.length === 0) return;
    patch({ stages: [...spec.stages, newStage(options.features[0])] });
  };

  const removeStage = (i: number) => {
    patch({ stages: spec.stages.filter((_, k) => k !== i) });
  };

  return (
    <div className="space-y-4 border-t border-gray-100 pt-4">
      <div className="space-y-1.5">
        <Label htmlFor="combinator">Combine stages by</Label>
        <select
          id="combinator"
          value={spec.combinator}
          disabled={disabled}
          onChange={(e) => patch({ combinator: e.target.value })}
          className={selectClass}
        >
          {options.combinators.map((c) => (
            <option key={c.name} value={c.name}>
              {c.label}
            </option>
          ))}
        </select>
      </div>

      <label className="flex items-center gap-2 text-sm text-gray-700">
        <input
          type="checkbox"
          checked={spec.short_circuit}
          disabled={disabled}
          onChange={(e) => patch({ short_circuit: e.target.checked })}
          className="h-4 w-4 rounded border-gray-300 accent-yellow-400"
        />
        Stop at the first deciding stage
      </label>

      <div className="space-y-3">
        {spec.stages.map((stage, i) => {
          const feat = featureOption(options, stage.feature);
          const rule = ruleOf(feat, stage.rule);
          return (
            <div key={i} className="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Stage {i + 1}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    className="p-1 text-gray-400 hover:text-gray-700 disabled:opacity-30"
                    disabled={disabled || i === 0}
                    onClick={() => move(i, -1)}
                    aria-label="Move up"
                  >
                    <ArrowUp className="h-3.5 w-3.5" />
                  </button>
                  <button
                    className="p-1 text-gray-400 hover:text-gray-700 disabled:opacity-30"
                    disabled={disabled || i === spec.stages.length - 1}
                    onClick={() => move(i, 1)}
                    aria-label="Move down"
                  >
                    <ArrowDown className="h-3.5 w-3.5" />
                  </button>
                  <button
                    className="p-1 text-gray-400 hover:text-red-600 disabled:opacity-30"
                    disabled={disabled || spec.stages.length <= 1}
                    onClick={() => removeStage(i)}
                    aria-label="Remove stage"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              <select
                value={stage.feature}
                disabled={disabled}
                onChange={(e) => {
                  const nextFeat = featureOption(options, e.target.value);
                  if (nextFeat) setStage(i, newStage(nextFeat));
                }}
                className={selectClass}
              >
                {options.features.map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.label}
                  </option>
                ))}
              </select>

              {feat && feat.rules.length > 0 && (
                <select
                  value={stage.rule}
                  disabled={disabled}
                  onChange={(e) => {
                    const nextRule = ruleOf(feat, e.target.value);
                    setStage(i, { ...stage, rule: e.target.value, params: seedStageParams(nextRule) });
                  }}
                  className={selectClass}
                >
                  {feat.rules.map((r) => (
                    <option key={r.name} value={r.name}>
                      {r.label}
                    </option>
                  ))}
                </select>
              )}

              <ThresholdRow
                label={feat?.threshold_label ?? "Threshold"}
                value={stage.threshold}
                disabled={disabled}
                onChange={(v) => setStage(i, { ...stage, threshold: v })}
              />

              {rule?.params.map((p) => (
                <ParamField
                  key={p.name}
                  spec={p}
                  value={stage.params[p.name]}
                  labels={feat?.labels ?? {}}
                  disabled={disabled}
                  onChange={(v) => setStage(i, { ...stage, params: { ...stage.params, [p.name]: v } })}
                />
              ))}
            </div>
          );
        })}
      </div>

      <Button size="sm" variant="ghost" className="w-full gap-1.5" disabled={disabled} onClick={addStage}>
        <Plus className="h-4 w-4" /> Add stage
      </Button>
    </div>
  );
}

function ThresholdRow({
  label,
  value,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <span className="text-xs font-mono font-medium text-yellow-600">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-yellow-400"
      />
    </div>
  );
}
