import { Upload } from "lucide-react";
import { Label } from "./ui/label";
import { ThresholdControl } from "./ThresholdControl";

interface Props {
  skus: string[];
  selectedSku: string | null;
  threshold: number;
  customerName: string;
  modelLoaded: boolean;
  onSkuChange: (sku: string) => void;
  onThresholdChange: (value: number) => void;
  onCustomerChange: (value: string) => void;
  onUpload: (file: File) => void;
}

export function InferencePanel({
  skus,
  selectedSku,
  threshold,
  customerName,
  modelLoaded,
  onSkuChange,
  onThresholdChange,
  onCustomerChange,
  onUpload,
}: Props) {
  const canUpload = modelLoaded && !!selectedSku;

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="space-y-1.5">
        <Label htmlFor="customer-name">Customer Name</Label>
        <input
          id="customer-name"
          type="text"
          value={customerName}
          onChange={(e) => onCustomerChange(e.target.value)}
          placeholder="Optional"
          className="flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="sku">SKU</Label>
        <select
          id="sku"
          value={selectedSku ?? ""}
          onChange={(e) => onSkuChange(e.target.value)}
          disabled={skus.length === 0}
          className="flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="" disabled>
            {skus.length ? "Select a SKU…" : "No SKUs found"}
          </option>
          {skus.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <p className="text-[11px] text-gray-400">
          Pick a SKU, then click any image to inspect it.
        </p>
      </div>

      <ThresholdControl value={threshold} onChange={onThresholdChange} />

      {!modelLoaded && (
        <p className="text-xs text-amber-600">
          Load a model via the ⚙ settings before inferring.
        </p>
      )}

      <div className="space-y-1.5 border-t border-gray-100 pt-4">
        <Label htmlFor="image-upload">Upload your own</Label>
        <label
          htmlFor="image-upload"
          className={`flex flex-col items-center justify-center w-full h-24 border-2 border-dashed rounded-lg transition-colors text-gray-400 ${
            canUpload
              ? "border-gray-300 cursor-pointer hover:border-blue-400 hover:bg-blue-50"
              : "border-gray-200 cursor-not-allowed opacity-60"
          }`}
        >
          <Upload className="h-5 w-5" />
          <span className="text-xs mt-1">Click to upload</span>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            className="hidden"
            disabled={!canUpload}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              e.target.value = "";
            }}
          />
        </label>
        <p className="text-[11px] text-gray-400">
          Infers immediately, tagged with the selected SKU.
        </p>
      </div>
    </div>
  );
}
