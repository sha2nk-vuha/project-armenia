import { useState } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { FileText } from "lucide-react";
import { api } from "../api/client";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";

interface Props {
  customerName: string;
}

export function ReportModal({ customerName }: Props) {
  const [open, setOpen] = useState(false);
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toIso(d: Date): string {
    return d.toISOString().replace("Z", "");
  }

  async function handleGenerate() {
    if (!startDate || !endDate) {
      setError("Select both a start and end date.");
      return;
    }
    if (endDate < startDate) {
      setError("End date must be after start date.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const blob = await api.generateReport(toIso(startDate), toIso(endDate), customerName);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const start = startDate.toISOString().slice(0, 10);
      const end = endDate.toISOString().slice(0, 10);
      a.download = `inspection-report-${start}-to-${end}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to generate report.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1.5">
          <FileText className="h-4 w-4" />
          Report
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Generate Inspection Report</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            {customerName ? `Report for: ${customerName}` : "Report for: All customers"}
          </p>

          <div className="space-y-1.5">
            <Label>Start Date</Label>
            <DatePicker
              selected={startDate}
              onChange={(d: Date | null) => setStartDate(d)}
              dateFormat="yyyy-MM-dd"
              placeholderText="Select start date"
              className="flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="space-y-1.5">
            <Label>End Date</Label>
            <DatePicker
              selected={endDate}
              onChange={(d: Date | null) => setEndDate(d)}
              dateFormat="yyyy-MM-dd"
              placeholderText="Select end date"
              minDate={startDate ?? undefined}
              className="flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {error && <p className="text-xs text-red-600">{error}</p>}

          <Button onClick={handleGenerate} disabled={loading} className="w-full">
            {loading ? "Generating…" : "Download PDF"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
