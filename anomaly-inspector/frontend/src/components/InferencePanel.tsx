import { FolderUp } from "lucide-react";
import { Label } from "./ui/label";
import { ThresholdControl } from "./ThresholdControl";
import { DecisionRuleControl } from "./DecisionRuleControl";
import { CascadeBuilder } from "./CascadeBuilder";
import type {
  CalibrationState,
  CascadeOptions,
  CascadeSpec,
  DecisionRuleInfo,
  FeatureInfo,
  ModelInfo,
  RuleParams,
} from "../api/client";

interface Props {
  selectedSku: string;
  threshold: number;
  customerName: string;
  modelLoaded: boolean;
  features: FeatureInfo[];
  activeFeature: string | null;
  featureSwitching: boolean;
  thresholdLabel: string;
  decisionRules: DecisionRuleInfo[];
  classLabels: Record<string, string>;
  selectedRule: string | null;
  ruleParams: RuleParams;
  calibration: CalibrationState | null;
  calibrationGroups: string[];
  calibrationGroup: string | null;
  calibrating: boolean;
  isCascade: boolean;
  cascadeOptions: CascadeOptions | null;
  cascadeSpec: CascadeSpec | null;
  loadedModels: Record<string, ModelInfo | undefined>;
  uploadingFeature: string | null;
  onUploadModel: (feature: string, model: File, version: string, sidecar: File | null) => void;
  onCascadeChange: (spec: CascadeSpec) => void;
  onRuleChange: (rule: string) => void;
  onRuleParamChange: (name: string, value: unknown) => void;
  onCalibrationGroupChange: (group: string) => void;
  onCalibrate: () => void;
  onClearCalibration: () => void;
  onFeatureChange: (feature: string) => void;
  onSkuChange: (sku: string) => void;
  onThresholdChange: (value: number) => void;
  onCustomerChange: (value: string) => void;
  onUploadDirectory: (files: FileList) => void;
  onRunAll: () => void;
  batchRunning: boolean;
}

export function InferencePanel({
  selectedSku,
  threshold,
  customerName,
  modelLoaded,
  features,
  activeFeature,
  featureSwitching,
  thresholdLabel,
  decisionRules,
  classLabels,
  selectedRule,
  ruleParams,
  calibration,
  calibrationGroups,
  calibrationGroup,
  calibrating,
  isCascade,
  cascadeOptions,
  cascadeSpec,
  loadedModels,
  uploadingFeature,
  onUploadModel,
  onCascadeChange,
  onRuleChange,
  onRuleParamChange,
  onCalibrationGroupChange,
  onCalibrate,
  onClearCalibration,
  onFeatureChange,
  onSkuChange,
  onThresholdChange,
  onCustomerChange,
  onUploadDirectory,
  onRunAll,
  batchRunning,
}: Props) {
  const canUpload = (isCascade || modelLoaded) && !!selectedSku;

  // `webkitdirectory`/`directory` are non-standard input attributes not present
  // in React's typings, so they're spread in as untyped props.
  const directoryProps = {
    webkitdirectory: "",
    directory: "",
    multiple: true,
  } as Record<string, unknown>;

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="space-y-1.5">
        <Label htmlFor="feature">Feature</Label>
        <select
          id="feature"
          value={activeFeature ?? ""}
          onChange={(e) => onFeatureChange(e.target.value)}
          disabled={features.length === 0 || featureSwitching}
          className="flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-yellow-400 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="" disabled>
            {features.length ? "Select a feature…" : "No features"}
          </option>
          {features.map((f) => (
            <option key={f.name} value={f.name}>
              {f.label}
            </option>
          ))}
        </select>
        <p className="text-[11px] text-gray-400">
          {featureSwitching ? "Switching feature…" : "Switching loads that feature's model."}
        </p>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="customer-name">Customer Name</Label>
        <input
          id="customer-name"
          type="text"
          value={customerName}
          onChange={(e) => onCustomerChange(e.target.value)}
          placeholder="Optional"
          className="flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-yellow-400"
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="sku">SKU</Label>
        <input
          id="sku"
          type="text"
          value={selectedSku}
          onChange={(e) => onSkuChange(e.target.value)}
          placeholder="Enter SKU name"
          className="flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-yellow-400"
        />
        <p className="text-[11px] text-gray-400">
          Enter a SKU name to scope calibration and stats.
        </p>
      </div>

      {isCascade ? (
        cascadeOptions && cascadeSpec ? (
          <CascadeBuilder
            options={cascadeOptions}
            spec={cascadeSpec}
            disabled={featureSwitching}
            loadedModels={loadedModels}
            uploadingFeature={uploadingFeature}
            onUploadModel={onUploadModel}
            onChange={onCascadeChange}
          />
        ) : (
          <p className="text-[11px] text-gray-400 border-t border-gray-100 pt-4">
            Loading cascade options…
          </p>
        )
      ) : (
        <>
          <ThresholdControl value={threshold} onChange={onThresholdChange} label={thresholdLabel} />

          <DecisionRuleControl
            rules={decisionRules}
            labels={classLabels}
            selected={selectedRule}
            params={ruleParams}
            disabled={featureSwitching}
            calibration={calibration}
            calibrationGroups={calibrationGroups}
            calibrationGroup={calibrationGroup}
            calibrating={calibrating}
            canCalibrate={!!selectedSku && calibrationGroups.length > 0}
            onSelect={onRuleChange}
            onParamChange={onRuleParamChange}
            onCalibrationGroupChange={onCalibrationGroupChange}
            onCalibrate={onCalibrate}
            onClearCalibration={onClearCalibration}
          />
        </>
      )}

      {!isCascade && !modelLoaded && (
        <p className="text-xs text-amber-600">
          Upload a model via the ⚙ Model button before inferring.
        </p>
      )}

      <div className="space-y-1.5 border-t border-gray-100 pt-4">
        <Label htmlFor="directory-upload">Upload a folder</Label>
        <label
          htmlFor="directory-upload"
          className={`flex flex-col items-center justify-center w-full h-24 border-2 border-dashed rounded-lg transition-colors text-gray-400 ${
            canUpload
              ? "border-gray-300 cursor-pointer hover:border-yellow-400 hover:bg-yellow-50"
              : "border-gray-200 cursor-not-allowed opacity-60"
          }`}
        >
          <FolderUp className="h-5 w-5" />
          <span className="text-xs mt-1">Click to select a folder</span>
          <input
            id="directory-upload"
            type="file"
            accept="image/*"
            className="hidden"
            disabled={!canUpload}
            {...directoryProps}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) onUploadDirectory(files);
              e.target.value = "";
            }}
          />
        </label>
        <p className="text-[11px] text-gray-400">
          Browse the folder's images, then click any to inspect it.
        </p>
      </div>

      <button
        type="button"
        onClick={onRunAll}
        disabled={!canUpload || batchRunning}
        className={`flex items-center justify-center gap-2 h-9 w-full rounded-md text-sm font-medium transition-colors ${
          canUpload && !batchRunning
            ? "bg-yellow-400 text-gray-900 hover:bg-yellow-500"
            : "bg-gray-200 text-gray-400 cursor-not-allowed"
        }`}
      >
        {batchRunning ? "Running…" : "Run all images"}
      </button>
    </div>
  );
}
