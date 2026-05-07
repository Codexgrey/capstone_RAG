import React, { useRef, useState } from "react";
import { X, Upload, Sliders } from "lucide-react";
import { uploadDocument } from "../services/documentService";
import toast from "react-hot-toast";

interface SidebarProps {
  open: boolean;
  setOpen: (v: boolean) => void;
  setMethod?: (method: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ open, setOpen, setMethod }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedMode, setSelectedMode] = useState("Vectors");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string[]>([]);

  const methodMap: Record<string, string> = {
    Vectors: "vector",
    Keyword: "keyword",
    Hybrid:  "hybrid",
  };

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const label = e.target.value;
    setSelectedMode(label);
    if (setMethod) setMethod(methodMap[label]);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setUploadProgress([]);

    const fileArray = Array.from(files);
    const results: string[] = [];

    for (const file of fileArray) {
      try {
        const result = await uploadDocument(file);
        results.push(`✅ ${result.filename}`);
        toast.success(`${result.filename} uploaded — processing in background`);
      } catch (error: any) {
        const status = error?.response?.status;
        const msg    = error?.response?.data?.detail || "Upload failed";
        if (status === 409) {
          results.push(`⚠️ ${file.name}: already in system`);
          toast.error(msg, { duration: 4000 });
        } else {
          results.push(`❌ ${file.name}: ${msg}`);
          toast.error(`${file.name}: ${msg}`);
        }
      }
    }

    setUploadProgress(results);
    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <>
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
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="size-5 text-primary" />
            <h2 className="text-lg font-bold text-base-content">Controls</h2>
          </div>
          <button className="btn btn-ghost btn-circle btn-sm" onClick={() => setOpen(false)}>
            <X className="size-4" />
          </button>
        </div>

        <div className="divider my-0" />

        {/* Retrieval Mode */}
        <div className="flex flex-col gap-2">
          <label className="text-xs uppercase tracking-widest text-base-content/60 font-semibold">
            Retrieval Mode
          </label>
          <select className="select select-bordered w-full" value={selectedMode} onChange={handleModeChange}>
            <option>Keyword</option>
            <option>Vectors</option>
            <option>Hybrid</option>
          </select>
        </div>

        <div className="divider my-0" />

        {/* Ingest Documents */}
        <div className="flex flex-col gap-3">
          <label className="text-xs uppercase tracking-widest text-base-content/60 font-semibold">
            Ingest Documents
          </label>

          <button
            className="btn btn-primary btn-outline w-full gap-2"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {uploading ? <span className="loading loading-spinner loading-xs" /> : <Upload className="size-4" />}
            {uploading ? "Uploading..." : "Upload"}
          </button>

          <input
            type="file"
            accept=".pdf,.txt,.md,.docx"
            multiple
            ref={fileInputRef}
            style={{ display: "none" }}
            onChange={handleFileUpload}
          />

          <p className="text-xs text-base-content/40 text-center">
            PDF · TXT · MD · DOCX <br /> Multiple files supported
          </p>

          {uploadProgress.length > 0 && (
            <div className="flex flex-col gap-1 mt-1">
              {uploadProgress.map((msg, i) => (
                <p key={i} className="text-xs text-base-content/60 truncate" title={msg}>{msg}</p>
              ))}
            </div>
          )}
        </div>

        <div className="divider my-0" />

        {/* Guardrails */}
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase tracking-widest text-base-content/60 font-semibold">
            Guardrails
          </label>
          <input type="checkbox" className="toggle toggle-primary" defaultChecked />
        </div>
      </div>

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
