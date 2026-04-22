import React, { useRef, useState } from "react";
import { uploadDocument } from "../services/documentService";

interface SidebarProps {
  open: boolean;
  setOpen: (v: boolean) => void;
  setMethod?: (method: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  open,
  setOpen,
  setMethod
}) => {
  /* ================= FILE INPUT REFS ================= */
  const pdfInputRef = useRef<HTMLInputElement>(null);
  const textInputRef = useRef<HTMLInputElement>(null);

  /* ================= DROPDOWN STATE ================= */
  const [selectedMode, setSelectedMode] =
    useState("Keyword");

  /* ================= MAP UI LABELS TO BACKEND VALUES ================= */
  const methodMap: Record<string, string> = {
    Vectors: "vector",
    Keyword: "keyword",
    CLARA: "clara"
  };

  /* ================= HANDLE MODE CHANGE ================= */
  const handleModeChange = (
    e: React.ChangeEvent<HTMLSelectElement>
  ) => {
    const label = e.target.value;

    setSelectedMode(label);

    const backendValue = methodMap[label];

    if (setMethod) {
      setMethod(backendValue);
    }
  };

  /* ================= PDF UPLOAD ================= */
  const handlePdfUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    if (!file) return;

    try {
      const result = await uploadDocument(file);

      console.log(
        "Uploaded PDF:",
        result.document_id
      );

      alert("PDF uploaded successfully");
    } catch (error) {
      console.error(error);
      alert("Upload failed");
    }
  };

  /* ================= TEXT UPLOAD ================= */
  const handleTextUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    if (!file) return;

    try {
      const result = await uploadDocument(file);

      console.log(
        "Uploaded Text:",
        result.document_id
      );

      alert("Text uploaded successfully");
    } catch (error) {
      console.error(error);
      alert("Upload failed");
    }
  };

  return (
    <>
      <div className={`sidebar ${open ? "open" : ""}`}>
        {/* CLOSE */}
        <button
          className="close-btn"
          onClick={() => setOpen(false)}
        >
          ✕
        </button>

        <h2>Controls</h2>

        {/* ================= RETRIEVAL MODE ================= */}
        <label>Retrieval Mode</label>

        <select
          value={selectedMode}
          onChange={handleModeChange}
        >
          <option>Keyword</option>
          <option>Vectors</option>
          <option>CLARA</option>
        </select>

        {/* ================= UPLOAD ================= */}
        <div className="upload">
          <button
            onClick={() =>
              pdfInputRef.current?.click()
            }
          >
            Upload PDF
          </button>

          <button
            onClick={() =>
              textInputRef.current?.click()
            }
          >
            Upload Text
          </button>

          {/* Hidden Inputs */}
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

        {/* ================= TOGGLE ================= */}
        <div className="toggle">
          Guardrails{" "}
          <input
            type="checkbox"
            defaultChecked
          />
        </div>
      </div>

      {/* OVERLAY */}
      {open && (
        <div
          className="overlay"
          onClick={() => setOpen(false)}
        />
      )}
    </>
  );
};

export default Sidebar;