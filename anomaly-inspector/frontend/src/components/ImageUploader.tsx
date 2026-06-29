import { Upload } from "lucide-react";
import { Label } from "./ui/label";

interface Props {
  onImageSelected: (file: File, preview: string) => void;
  preview: string | null;
}

export function ImageUploader({ onImageSelected, preview }: Props) {
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    onImageSelected(file, url);
  }

  return (
    <div className="space-y-1.5">
      <Label htmlFor="image-upload">Image</Label>
      <label
        htmlFor="image-upload"
        className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors overflow-hidden"
      >
        {preview ? (
          <img src={preview} alt="preview" className="h-full w-full object-contain" />
        ) : (
          <div className="flex flex-col items-center gap-1 text-gray-400">
            <Upload className="h-6 w-6" />
            <span className="text-xs">Click to upload image</span>
          </div>
        )}
        <input
          id="image-upload"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />
      </label>
    </div>
  );
}
