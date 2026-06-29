import { Zap } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { ImageUploader } from "./ImageUploader";
import { ThresholdControl } from "./ThresholdControl";

interface Props {
  skuName: string;
  modelLoaded: boolean;
  selectedImage: File | null;
  imagePreview: string | null;
  threshold: number;
  isInferring: boolean;
  onSkuChange: (value: string) => void;
  onImageSelected: (file: File, preview: string) => void;
  onThresholdChange: (value: number) => void;
  onInfer: () => void;
}

export function InferencePanel({
  skuName, modelLoaded, selectedImage, imagePreview,
  threshold, isInferring, onSkuChange, onImageSelected,
  onThresholdChange, onInfer,
}: Props) {
  const canInfer = modelLoaded && selectedImage !== null && skuName.trim().length > 0;

  return (
    <div className="flex flex-col gap-4 p-4 h-full">
      <div className="space-y-1.5">
        <Label htmlFor="sku">SKU Name</Label>
        <Input
          id="sku"
          placeholder="e.g. WIDGET-001"
          value={skuName}
          onChange={(e) => onSkuChange(e.target.value)}
        />
      </div>

      <ImageUploader onImageSelected={onImageSelected} preview={imagePreview} />

      <ThresholdControl value={threshold} onChange={onThresholdChange} />

      {!modelLoaded && (
        <p className="text-xs text-amber-600">
          Load a model via the ⚙ settings before inferring.
        </p>
      )}

      <Button
        onClick={onInfer}
        disabled={!canInfer || isInferring}
        size="lg"
        className="mt-auto gap-2"
      >
        <Zap className="h-4 w-4" />
        {isInferring ? "Analysing…" : "Infer"}
      </Button>
    </div>
  );
}
