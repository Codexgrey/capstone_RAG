import React from "react";
import { FileText, Clock, Zap, Database } from "lucide-react";

interface Citation {
  chunk_id:        string;
  source_name:     string;
  source?:         string;
  document_title?: string;
  page:            number;
  section:         string | null;
}

interface SourcesPanelProps {
  citations:        Citation[];
  latency_ms:       number;
  retrieval_method?: string;
}

const METHOD_LABELS: Record<string, { label: string; color: string }> = {
  vector:  { label: "Vector (FAISS)",       color: "badge-primary"  },
  keyword: { label: "Keyword (BM25)",       color: "badge-secondary"},
  hybrid:  { label: "Hybrid (FAISS+BM25)",  color: "badge-accent"   },
  none:    { label: "No retrieval",         color: "badge-ghost"    },
};

const SourcesPanel: React.FC<SourcesPanelProps> = ({ citations, latency_ms, retrieval_method }) => {
  const methodInfo = retrieval_method ? (METHOD_LABELS[retrieval_method] ?? { label: retrieval_method, color: "badge-ghost" }) : null;

  return (
    <div className="flex flex-col gap-4">

      {/* Sources */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body gap-3">
          <h3 className="text-xs uppercase tracking-widest text-base-content/60 font-semibold flex items-center gap-2">
            <FileText className="size-4" />
            Sources
          </h3>

          {methodInfo && (
            <div className="flex items-center gap-2">
              <Database className="size-3 text-base-content/40" />
              <span className={`badge badge-sm ${methodInfo.color}`}>{methodInfo.label}</span>
            </div>
          )}

          {citations.length === 0 ? (
            <p className="text-base-content/40 text-sm italic">
              No sources yet — ask a question first
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {citations.map((c, i) => {
                const displayName = c.document_title || c.source_name || c.source || "Unknown";
                const fileName    = c.source_name || c.source || "Unknown";
                return (
                  <div key={i} className="flex items-start gap-2 bg-base-200 rounded-lg p-3">
                    <span className="badge badge-primary badge-sm mt-0.5 shrink-0">{i + 1}</span>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-base-content text-sm font-medium">{displayName}</span>
                      <span className="text-base-content/50 text-xs">
                        {fileName !== displayName && `${fileName} · `}
                        {c.page ? `Page ${c.page}` : ""}
                        {c.section ? ` · ${c.section}` : ""}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Performance */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body gap-3">
          <h3 className="text-xs uppercase tracking-widest text-base-content/60 font-semibold flex items-center gap-2">
            <Zap className="size-4" />
            Performance
          </h3>
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <span className="text-base-content/60 text-sm flex items-center gap-1">
                <Clock className="size-3" /> Total latency
              </span>
              <span className="badge badge-outline">
                {latency_ms > 0 ? `${latency_ms.toFixed(0)} ms` : "--"}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-base-content/60 text-sm">Guardrail</span>
              <span className="badge badge-success badge-outline">active</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default SourcesPanel;
