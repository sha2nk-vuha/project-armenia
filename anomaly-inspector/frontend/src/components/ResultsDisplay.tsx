import { useEffect, useRef, useState } from "react";

interface ImagePanelProps {
  src: string | null;
  label: string;
}

function ImagePanel({ src, label }: ImagePanelProps) {
  return (
    <div className="flex flex-col gap-1 flex-1 min-w-0">
      <span className="text-xs font-medium text-gray-500 text-center">{label}</span>
      <div className="aspect-square w-full rounded-lg bg-gray-100 border border-gray-200 overflow-hidden flex items-center justify-center">
        {src ? (
          <img src={src} alt={label} className="w-full h-full object-contain" />
        ) : (
          <span className="text-xs text-gray-400">—</span>
        )}
      </div>
    </div>
  );
}

interface Props {
  originalPreview: string | null;
  heatmap: string | null;
  segmentation: string | null;
}

// Below this container width the three panels stack vertically; at or above it
// they sit side by side. ~600px gives each of the three panels a usable size.
const HORIZONTAL_BREAKPOINT = 600;

export function ResultsDisplay({ originalPreview, heatmap, segmentation }: Props) {
  const heatmapSrc = heatmap ? `data:image/jpeg;base64,${heatmap}` : null;
  const segSrc = segmentation ? `data:image/jpeg;base64,${segmentation}` : null;

  // Reflow based on the container's own width (works for the resizable pane).
  const ref = useRef<HTMLDivElement>(null);
  const [horizontal, setHorizontal] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setHorizontal(entry.contentRect.width >= HORIZONTAL_BREAKPOINT);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`flex gap-3 w-full ${horizontal ? "flex-row items-start" : "flex-col"}`}
    >
      <ImagePanel src={originalPreview} label="Original" />
      <ImagePanel src={heatmapSrc} label="Heatmap" />
      <ImagePanel src={segSrc} label="Segmentation" />
    </div>
  );
}
