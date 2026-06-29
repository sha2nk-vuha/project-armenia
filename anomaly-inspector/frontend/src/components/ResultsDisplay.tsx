interface ImagePanelProps {
  src: string | null;
  label: string;
}

function ImagePanel({ src, label }: ImagePanelProps) {
  return (
    <div className="flex flex-col gap-1 flex-1">
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

export function ResultsDisplay({ originalPreview, heatmap, segmentation }: Props) {
  const heatmapSrc = heatmap ? `data:image/jpeg;base64,${heatmap}` : null;
  const segSrc = segmentation ? `data:image/jpeg;base64,${segmentation}` : null;

  return (
    <div className="flex gap-3 w-full">
      <ImagePanel src={originalPreview} label="Original" />
      <ImagePanel src={heatmapSrc} label="Heatmap" />
      <ImagePanel src={segSrc} label="Segmentation" />
    </div>
  );
}
