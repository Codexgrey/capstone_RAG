import React from "react";

/* ================= TYPES ================= */
interface Citation {
  chunk_id: string;
  source_name: string;
  page: number;
  section: string | null;
}

interface SourcesPanelProps {
  citations: Citation[];
  latency_ms: number;
}

/* ================= COMPONENT ================= */
const SourcesPanel: React.FC<SourcesPanelProps> = ({
  citations,
  latency_ms
}) => {
  return (
    <aside className="right-panel">
      <h3>Sources</h3>

      <ul className="sources">
        {citations.length === 0 ? (
          <li>No sources yet</li>
        ) : (
          citations.map((c, i) => (
            <li key={i}>
              [{i + 1}] {c.source_name} — Page {c.page}
            </li>
          ))
        )}
      </ul>

      <div className="metrics">
        <h3>Performance</h3>

        <p>Retrieval: {latency_ms} ms</p>
        <p>LLM: --</p>
        <p>Guardrail: --</p>
      </div>
    </aside>
  );
};

export default SourcesPanel;