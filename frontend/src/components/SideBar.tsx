import React, { useRef, useState } from "react";
import { X, Upload, FileText, Sliders } from "lucide-react";
import { uploadDocument } from "../services/documentService";
import toast from "react-hot-toast";

interface SidebarProps {
  open: boolean;
  setOpen: (v: boolean) => void;
  setMethod?: (method: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ open, setOpen, setMethod }) => {
  const pdfInputRef  = useRef<HTMLInputElement>(null);
  const textInputRef = useRef<HTMLInputElement>(null);
  const [selectedMode, setSelectedMode] = useState("Vectors");
  const [uploading, setUploading]       = useState(false);

  /* ── Map UI labels to backend values ── */
  const methodMap: Record<string, string> = {
    Vectors: "vector",
    Keyword: "keyword",
    Hybrid:  "hybrid",
  };

  /* ── Handle retrieval mode change ── */
  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const label = e.target.value;
    setSelectedMode(label);
    if (setMethod) setMethod(methodMap[label]);
  };

  /* ── PDF Upload — uses documentService.uploadDocument() ── */
  const handlePdfUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      toast.success(`${result.filename} uploaded — processing in background`);
    } catch (error: any) {
      const msg = error?.response?.data?.detail || "Upload failed";
      toast.error(msg);
    } finally {
      setUploading(false);
      // Reset input so same file can be re-uploaded
      if (pdfInputRef.current) pdfInputRef.current.value = "";
    }
  };

  /* ── Text Upload — uses documentService.uploadDocument() ── */
  const handleTextUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      toast.success(`${result.filename} uploaded — processing in background`);
    } catch (error: any) {
      const msg = error?.response?.data?.detail || "Upload failed";
      toast.error(msg);
    } finally {
      setUploading(false);
      if (textInputRef.current) textInputRef.current.value = "";
    }
  };

  return (
    <>
      {/* ── Slide-in sidebar (keep animation via inline style) ── */}
      <div
        style={{
          position:   "fixed",
          top:        0,
          left:       0,
          height:     "100%",
          width:      "280px",
          zIndex:     50,
          transform:  open ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 0.3s ease",
        }}
        className="bg-base-300 border-r border-base-content/10 flex flex-col p-6 gap-6 overflow-y-auto"
      >
        {/* Header row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="size-5 text-primary" />
            <h2 className="text-lg font-bold text-base-content">Controls</h2>
          </div>
          <button
            className="btn btn-ghost btn-circle btn-sm"
            onClick={() => setOpen(false)}
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="divider my-0" />

        {/* ── Retrieval Mode ── */}
        <div className="flex flex-col gap-2">
          <label className="text-xs uppercase tracking-widest text-base-content/60 font-semibold">
            Retrieval Mode
          </label>
          <select
            className="select select-bordered w-full"
            value={selectedMode}
            onChange={handleModeChange}
          >
            <option>Keyword</option>
            <option>Vectors</option>
            <option>Hybrid</option>
          </select>
        </div>

        <div className="divider my-0" />

        {/* ── Upload ── */}
        <div className="flex flex-col gap-3">
          <label className="text-xs uppercase tracking-widest text-base-content/60 font-semibold">
            Upload Documents
          </label>

          <button
            className="btn btn-primary btn-outline w-full gap-2"
            disabled={uploading}
            onClick={() => pdfInputRef.current?.click()}
          >
            {uploading ? (
              <span className="loading loading-spinner loading-xs" />
            ) : (
              <Upload className="size-4" />
            )}
            Upload PDF
          </button>

          <button
            className="btn btn-primary btn-outline w-full gap-2"
            disabled={uploading}
            onClick={() => textInputRef.current?.click()}
          >
            {uploading ? (
              <span className="loading loading-spinner loading-xs" />
            ) : (
              <FileText className="size-4" />
            )}
            Upload Text
          </button>

          {/* Hidden file inputs */}
          <input
            type="file"
            accept="application/pdf"
            ref={pdfInputRef}
            style={{ display: "none" }}
            onChange={handlePdfUpload}
          />
          <input
            type="file"
            accept=".txt"
            ref={textInputRef}
            style={{ display: "none" }}
            onChange={handleTextUpload}
          />
        </div>

        <div className="divider my-0" />

        {/* ── Guardrails Toggle ── */}
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase tracking-widest text-base-content/60 font-semibold">
            Guardrails
          </label>
          <input type="checkbox" className="toggle toggle-primary" defaultChecked />
        </div>
      </div>

      {/* ── Overlay ── */}
      {open && (
        <div
          style={{ position: "fixed", inset: 0, zIndex: 40, background: "rgba(0,0,0,0.5)" }}
          onClick={() => setOpen(false)}
        />
      )}
    </>
  );
};

export default Sidebar;