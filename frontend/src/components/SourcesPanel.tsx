import React from "react";
import { FileText, Zap, Database, BarChart2, Info, Award } from "lucide-react";

interface Citation {
  chunk_id:        string;
  source_name:     string;
  source?:         string;
  document_title?: string;
  page:            number;
  section:         string | null;
}

interface EvaluationMetrics {
  // Retrieval quality (always present)
  top_score?:        number;
  avg_score?:        number;
  source_coverage?:  number;
  chunks_retrieved?: number;
  precision_at_k?:   number;
  mrr?:              number;
  source_diversity?: number;
  // TriviaQA benchmark (present when question is in test bank)
  triviaqa_em?:      number;
  triviaqa_f1?:      number;
  triviaqa_qid?:     string;
}

interface SourcesPanelProps {
  citations:           Citation[];
  latency_ms:          number;
  retrieval_method?:   string;
  evaluation_metrics?: EvaluationMetrics;
}

const METHOD_LABELS: Record<string, { label: string; color: string }> = {
  vector:  { label: "Vector (FAISS)",      color: "badge-primary"   },
  keyword: { label: "Keyword (BM25)",      color: "badge-secondary" },
  hybrid:  { label: "Hybrid (FAISS+BM25)", color: "badge-accent"    },
  none:    { label: "No retrieval",        color: "badge-ghost"     },
};

/* Mini progress bar — value 0–1 */
function ScoreBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  const color =
    pct >= 60 ? "bg-success" :
    pct >= 35 ? "bg-warning" :
                "bg-error";
  return (
    <div className="flex items-center gap-1.5 flex-1">
      <div className="w-full bg-base-300 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-1.5 rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-base-content/70 w-8 text-right">{pct}%</span>
    </div>
  );
}

/* Single metric row */
function MetricRow({
  label, value, showBar = false, badge, tooltip,
}: {
  label:    string;
  value:    string | number;
  showBar?: boolean;
  badge?:   string;
  tooltip?: string;
}) {
  return (
    <div className="flex justify-between items-center gap-2" title={tooltip}>
      <span className="text-base-content/60 text-xs flex items-center gap-1 shrink-0">
        {label}
        {tooltip && <Info className="size-2.5 opacity-40" />}
      </span>
      {showBar && typeof value === "number" ? (
        <ScoreBar value={value} />
      ) : badge ? (
        <span className={`badge badge-sm ${badge}`}>{value}</span>
      ) : (
        <span className="badge badge-outline badge-sm font-mono">{value}</span>
      )}
    </div>
  );
}

/* TriviaQA benchmark score badge */
function TriviaQABadge({ em, f1, qid }: { em: number; f1: number; qid: string }) {
  const emPct = Math.round(em * 100);
  const f1Pct = Math.round(f1 * 100);
  const emColor = em === 1 ? "badge-success" : em > 0 ? "badge-warning" : "badge-error";
  return (
    <div
      className="rounded-lg bg-base-200 p-2.5 flex flex-col gap-1.5"
      title={`TriviaQA benchmark question ${qid}`}
    >
      <span className="text-xs font-semibold text-base-content/70 flex items-center gap-1">
        <Award className="size-3" />
        TriviaQA Benchmark
        <span className="badge badge-xs badge-ghost ml-auto font-mono">{qid}</span>
      </span>

      <div className="flex items-center gap-2 mt-0.5">
        <span className="text-xs text-base-content/50 w-7 shrink-0">EM</span>
        <span className={`badge badge-sm ${emColor} font-mono`}>{emPct}%</span>
        <span className="text-base-content/30 text-xs">exact match</span>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-base-content/50 w-7 shrink-0">F1</span>
        <ScoreBar value={f1} />
      </div>

      {em < 1 && (
        <p className="text-xs text-base-content/40 italic mt-0.5">
          {em === 0
            ? "Answer did not match any valid alias — may be correct but phrased differently."
            : "Partial token match — check phrasing against ground truth."}
        </p>
      )}
    </div>
  );
}

const SourcesPanel: React.FC<SourcesPanelProps> = ({
  citations,
  latency_ms,
  retrieval_method,
  evaluation_metrics,
}) => {
  const methodInfo = retrieval_method
    ? (METHOD_LABELS[retrieval_method] ?? { label: retrieval_method, color: "badge-ghost" })
    : null;

  const em = evaluation_metrics;
  const hasMetrics    = em && (em.chunks_retrieved ?? 0) > 0;
  const hasTriviaQA   = em && em.triviaqa_qid !== undefined;

  return (
    <div className="flex flex-col gap-4">

      {/* ── Sources ──────────────────────────────────────────────────── */}
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

      {/* ── Performance ───────────────────────────────────────────────── */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body gap-3">
          <h3 className="text-xs uppercase tracking-widest text-base-content/60 font-semibold flex items-center gap-2">
            <Zap className="size-4" />
            Performance
          </h3>
          <div className="flex flex-col gap-2">
            <MetricRow
              label="Total latency"
              value={latency_ms > 0 ? `${latency_ms.toFixed(0)} ms` : "--"}
              badge="badge-outline"
              tooltip="End-to-end response time including retrieval + generation"
            />
            <MetricRow
              label="Guardrail"
              value="active"
              badge="badge-success badge-outline"
              tooltip="Retrieval-constrained prompting active — LLM grounded to retrieved context only"
            />
          </div>
        </div>
      </div>

      {/* ── Evaluation Metrics ────────────────────────────────────────── */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body gap-3">
          <h3 className="text-xs uppercase tracking-widest text-base-content/60 font-semibold flex items-center gap-2">
            <BarChart2 className="size-4" />
            Evaluation Metrics
          </h3>

          {!hasMetrics ? (
            <p className="text-base-content/40 text-xs italic">
              Ask a question to compute retrieval metrics.
            </p>
          ) : (
            <div className="flex flex-col gap-2.5">

              {/* ── TriviaQA EM/F1 (shown first when available) ── */}
              {hasTriviaQA && (
                <>
                  <TriviaQABadge
                    em={em!.triviaqa_em ?? 0}
                    f1={em!.triviaqa_f1 ?? 0}
                    qid={em!.triviaqa_qid!}
                  />
                  <div className="divider my-0.5 text-xs text-base-content/30">retrieval</div>
                </>
              )}

              {/* Top score */}
              <div
                className="flex items-center gap-2"
                title="Similarity/relevance score of the best-ranked retrieved chunk"
              >
                <span className="text-base-content/60 text-xs shrink-0 flex items-center gap-1">
                  Top Score <Info className="size-2.5 opacity-40" />
                </span>
                <ScoreBar value={em!.top_score ?? 0} />
              </div>

              {/* Avg score */}
              <div
                className="flex items-center gap-2"
                title="Mean similarity/relevance score across all retrieved chunks"
              >
                <span className="text-base-content/60 text-xs shrink-0 flex items-center gap-1">
                  Avg Score <Info className="size-2.5 opacity-40" />
                </span>
                <ScoreBar value={em!.avg_score ?? 0} />
              </div>

              {/* Precision@k */}
              <div
                className="flex items-center gap-2"
                title="Fraction of retrieved chunks scoring ≥ 0.40 (relevance proxy)"
              >
                <span className="text-base-content/60 text-xs shrink-0 flex items-center gap-1">
                  Precision@k <Info className="size-2.5 opacity-40" />
                </span>
                <ScoreBar value={em!.precision_at_k ?? 0} />
              </div>

              {/* MRR */}
              <div
                className="flex items-center gap-2"
                title="Mean Reciprocal Rank — how early the first relevant chunk appears"
              >
                <span className="text-base-content/60 text-xs shrink-0 flex items-center gap-1">
                  MRR <Info className="size-2.5 opacity-40" />
                </span>
                <ScoreBar value={em!.mrr ?? 0} />
              </div>

              {/* Source diversity */}
              <div
                className="flex items-center gap-2"
                title="Fraction of retrieved chunks from distinct source documents"
              >
                <span className="text-base-content/60 text-xs shrink-0 flex items-center gap-1">
                  Src Diversity <Info className="size-2.5 opacity-40" />
                </span>
                <ScoreBar value={em!.source_diversity ?? 0} />
              </div>

              <div className="divider my-0.5" />

              <MetricRow
                label="Chunks retrieved"
                value={em!.chunks_retrieved ?? "--"}
                tooltip="Number of top-k chunks returned by the retriever"
              />
              <MetricRow
                label="Unique sources"
                value={em!.source_coverage ?? "--"}
                tooltip="Number of distinct documents represented in top-k results"
              />
            </div>
          )}
        </div>
      </div>

    </div>
  );
};

export default SourcesPanel;
