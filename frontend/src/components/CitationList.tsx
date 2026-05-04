import React from "react";
import { FileText, ExternalLink } from "lucide-react";

interface Citation {
  chunk_id:        string;
  source_name?:    string;
  source?:         string;
  document_title?: string;
  page?:           number;
  section?:        string | null;
  file_type?:      string;
}

interface CitationListProps {
  citations: Citation[];
  compact?:  boolean;   // compact=true shows inline badges, false shows full cards
}

const CitationList: React.FC<CitationListProps> = ({ citations, compact = false }) => {
  if (!citations || citations.length === 0) return null;

  if (compact) {
    return (
      <div className="flex flex-wrap gap-2 mt-2">
        {citations.map((c, i) => {
          const name = c.document_title || c.source_name || c.source || "Unknown";
          return (
            <span key={i} className="badge badge-outline badge-sm gap-1">
              <FileText className="size-3" />
              {name}{c.page ? ` · p.${c.page}` : ""}
            </span>
          );
        })}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {citations.map((c, i) => {
        const name     = c.document_title || c.source_name || c.source || "Unknown";
        const fileName = c.source_name || c.source || "";
        return (
          <div key={i} className="flex items-start gap-3 bg-base-200 rounded-lg p-3">
            <span className="badge badge-primary badge-sm mt-0.5 shrink-0">{i + 1}</span>
            <div className="flex flex-col gap-0.5 min-w-0">
              <span className="text-sm font-medium text-base-content truncate">{name}</span>
              <div className="flex items-center gap-2 text-xs text-base-content/50">
                {fileName && fileName !== name && (
                  <span className="flex items-center gap-1">
                    <FileText className="size-3" />{fileName}
                  </span>
                )}
                {c.page && <span>Page {c.page}</span>}
                {c.section && <span>· {c.section}</span>}
                {c.file_type && (
                  <span className="badge badge-ghost badge-xs uppercase">{c.file_type}</span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default CitationList;