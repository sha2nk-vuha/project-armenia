import * as SliderPrimitive from "@radix-ui/react-slider";
import { Label } from "./ui/label";

interface Props {
  value: number;
  onChange: (value: number) => void;
}

export function ThresholdControl({ value, onChange }: Props) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>Threshold</Label>
        <span className="text-sm font-mono font-medium text-yellow-600">
          {value.toFixed(2)}
        </span>
      </div>
      <div className="flex items-center gap-2 text-xs text-gray-400">
        <span>0.00</span>
        <SliderPrimitive.Root
          className="relative flex flex-1 touch-none select-none items-center"
          min={0}
          max={1}
          step={0.01}
          value={[value]}
          onValueChange={([v]) => onChange(v)}
        >
          <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-gray-200">
            <SliderPrimitive.Range className="absolute h-full bg-yellow-400" />
          </SliderPrimitive.Track>
          <SliderPrimitive.Thumb className="block h-4 w-4 rounded-full border-2 border-yellow-400 bg-white shadow transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-300" />
        </SliderPrimitive.Root>
        <span>1.00</span>
      </div>
    </div>
  );
}
