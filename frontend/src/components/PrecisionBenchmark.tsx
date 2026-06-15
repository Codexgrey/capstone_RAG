/**
 * PrecisionBenchmark.tsx
 * ======================
 * TriviaQA Precision/Recall benchmark panel — isolated per-question
 * evaluation against TriviaQA's own evidence documents.
 *
 * Unlike EvalBenchmark.tsx (which runs the full RAG pipeline against the
 * team's own ingested corpus and scores EM/F1), this panel runs each
 * question in isolation: that question's own evidence is chunked and
 * ingested into the chosen retrieval method, retrieve() is scored against
 * that evidence as the gold set, then the evidence is removed before the
 * next question starts. Precision@k and Recall@k measure how many of the
 * retrieved chunks actually came from the question's own evidence.
 *
 * Because each question rebuilds a small index from scratch, runs are
 * resumable in batches rather than all-at-once — the panel shows how many
 * questions have been scored so far and lets the user continue.
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  Target, Play, RotateCcw, XCircle, BarChart2, Filter,
} from "lucide-react";

// Re-use the API base URL already defined in queryService
const API_BASE = (import.meta as { env: { VITE_API_URL?: string } }).env?.VITE_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CheckpointSnapshot {
  questions_done:          number;
  n_scored:                number;
  answer_in_context_at_k:  number;
  answer_in_top1:          number;
  mrr:                     number;
  precision_at_k:          number;
  recall_at_k:             number;
  avg_latency_ms:          number;
}

interface PrecisionResult {
  method:                  string;
  resume_index:            number;
  this_batch_questions:    number;
  n_scored:                number;
  n_skipped_no_evidence:   number;
  top_k:                   number | null;
  chunk_size:              number | null;
  chunk_overlap:           number | null;
  answer_in_context_at_k:  number;
  answer_in_top1:          number;
  mrr:                     number;
  precision_at_k:          number;
  recall_at_k:             number;
  avg_latency_ms:          number;
  checkpoints:             CheckpointSnapshot[];
  scoring_protocol?:       string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const METHOD_LABELS: Record<string, string> = {
  vector:  "Vector (FAISS)",
  keyword: "Keyword (BM25)",
  hybrid:  "Hybrid (FAISS+BM25)",
};

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

// ---------------------------------------------------------------------------
// Summary cards after a run
// ---------------------------------------------------------------------------

function SummaryCard({ result }: { result: PrecisionResult }) {
  return (
    <div className="stats stats-horizontal shadow w-full bg-base-200 flex-wrap">
      <div className="stat">
        <div className="stat-title">Method</div>
        <div className="stat-value text-lg">{METHOD_LABELS[result.method] ?? result.method}</div>
      </div>
      <div className="stat">
        <div className="stat-title">Precision@{result.top_k ?? "k"}</div>
        <div className="stat-value text-success">{pct(result.precision_at_k)}</div>
        <div className="stat-desc">retrieved chunks that are gold</div>
      </div>
      <div className="stat">
        <div className="stat-title">Recall@{result.top_k ?? "k"}</div>
        <div className="stat-value text-info">{pct(result.recall_at_k)}</div>
        <div className="stat-desc">gold chunks retrieved</div>
      </div>
      <div className="stat">
        <div className="stat-title">Answer@{result.top_k ?? "k"}</div>
        <div className="stat-value">{pct(result.answer_in_context_at_k)}</div>
        <div className="stat-desc">Top-1: {pct(result.answer_in_top1)} · MRR: {result.mrr.toFixed(3)}</div>
      </div>
      <div className="stat">
        <div className="stat-title">Avg Latency</div>
        <div className="stat-value text-lg">{result.avg_latency_ms.toFixed(1)} ms</div>
        <div className="stat-desc">per question</div>
      </div>
      <div className="stat">
        <div className="stat-title">Questions Scored</div>
        <div className="stat-value">{result.n_scored.toLocaleString()}</div>
        <div className="stat-desc">
          {result.n_skipped_no_evidence > 0
            ? `${result.n_skipped_no_evidence} skipped (no evidence)`
            : "resume index " + result.resume_index.toLocaleString()}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Progress-across-the-run table
// ---------------------------------------------------------------------------

function CheckpointTable({ checkpoints, topK }: { checkpoints: CheckpointSnapshot[]; topK: number | null }) {
  if (!checkpoints.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="table table-xs">
        <thead>
          <tr>
            <th>Questions Done</th>
            <th>Precision@{topK ?? "k"}</th>
            <th>Recall@{topK ?? "k"}</th>
            <th>Answer@{topK ?? "k"}</th>
            <th>Top-1</th>
            <th>MRR</th>
            <th>Avg Latency</th>
          </tr>
        </thead>
        <tbody>
          {checkpoints.map(cp => (
            <tr key={cp.questions_done}>
              <td className="font-mono">{cp.questions_done.toLocaleString()}</td>
              <td>{pct(cp.precision_at_k)}</td>
              <td>{pct(cp.recall_at_k)}</td>
              <td>{pct(cp.answer_in_context_at_k)}</td>
              <td>{pct(cp.answer_in_top1)}</td>
              <td>{cp.mrr.toFixed(3)}</td>
              <td>{cp.avg_latency_ms.toFixed(1)} ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const PrecisionBenchmark: React.FC = () => {
  const [method,     setMethod]     = useState("vector");
  const [topK,       setTopK]       = useState(5);
  const [batchSize,  setBatchSize]  = useState(200);
  const [chunkSize,  setChunkSize]  = useState(200);
  const [chunkOverlap, setChunkOverlap] = useState(40);

  const [status,   setStatus]   = useState<PrecisionResult | null>(null);
  const [result,   setResult]   = useState<PrecisionResult | null>(null);
  const [running,  setRunning]  = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const token = () => localStorage.getItem("token") ?? "";

  const fetchStatus = useCallback(async (m: string) => {
    try {
      const res = await fetch(
        `${API_BASE}/api/evaluate/triviaqa-precision/status?retrieval_method=${m}`,
        { headers: { Authorization: `Bearer ${token()}` } },
      );
      if (!res.ok) return;
      const data = await res.json();
      setStatus(data);
    } catch {
      // status is best-effort — ignore failures
    }
  }, []);

  // Load resume status whenever the selected method changes
  useEffect(() => {
    setResult(null);
    setError(null);
    fetchStatus(method);
  }, [method, fetchStatus]);

  const runBenchmark = useCallback(async () => {
    setRunning(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/evaluate/triviaqa-precision/run`, {
        method:  "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization:  `Bearer ${token()}`,
        },
        body: JSON.stringify({
          retrieval_method: method,
          top_k:            topK,
          batch_size:       batchSize,
          chunk_size:       chunkSize,
          chunk_overlap:    chunkOverlap,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      const data: PrecisionResult = await res.json();
      setResult(data);
      setStatus(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setRunning(false);
    }
  }, [method, topK, batchSize, chunkSize, chunkOverlap]);

  const resetBenchmark = useCallback(async () => {
    setResetting(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/evaluate/triviaqa-precision/reset?retrieval_method=${method}`,
        {
          method:  "POST",
          headers: { Authorization: `Bearer ${token()}` },
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      const data: PrecisionResult = await res.json();
      setStatus(data);
      setResult(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setResetting(false);
    }
  }, [method]);

  const display = result ?? status;
  const resumeIndex = status?.resume_index ?? 0;

  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto p-4">

      {/* Header */}
      <div className="flex items-center gap-3">
        <Target className="size-6 text-primary" />
        <div>
          <h2 className="text-xl font-bold">TriviaQA Precision Benchmark</h2>
          <p className="text-base-content/50 text-sm">
            Isolated per-question evaluation — Precision@k / Recall@k against TriviaQA&apos;s own evidence
          </p>
        </div>
        {resumeIndex > 0 && (
          <span className="badge badge-primary ml-auto">
            {resumeIndex.toLocaleString()} questions scored so far
          </span>
        )}
      </div>

      {/* Info */}
      <div className="alert text-sm bg-success text-white border border-success/20 shadow-sm">
        <BarChart2 className="size-5 shrink-0" />
        <div>
          <strong>How this works:</strong> For each TriviaQA question, that question&apos;s own
          evidence documents are chunked and ingested into the chosen retrieval method in
          isolation. Retrieval is then scored two ways: Precision@k / Recall@k measure how many
          of the retrieved chunks actually belong to that question&apos;s evidence (the gold set),
          while Answer-in-Context@k / Top-1 / MRR measure whether a correct answer alias appears
          in the retrieved text. The evidence is removed before the next question starts, so
          questions never share an index. Runs are resumable in batches.
        </div>
      </div>

      {/* Controls */}
      <div className="card bg-base-100 shadow">
        <div className="card-body gap-4">
          <h3 className="card-title text-base flex items-center gap-2">
            <Filter className="size-4" /> Configure Run
          </h3>

          <div className="flex flex-wrap gap-4 items-end">

            {/* Method */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Retrieval Method</span>
              </label>
              <select
                className="select select-bordered select-sm w-48"
                value={method}
                onChange={e => setMethod(e.target.value)}
                disabled={running}
              >
                <option value="vector">Vector (FAISS)</option>
                <option value="keyword">Keyword (BM25)</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </div>

            {/* Top-k */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Top-K Chunks</span>
              </label>
              <select
                className="select select-bordered select-sm w-24"
                value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                disabled={running}
              >
                {[3, 5, 7, 10].map(k => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>

            {/* Batch size (questions this run) */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Questions This Run</span>
              </label>
              <select
                className="select select-bordered select-sm w-28"
                value={batchSize}
                onChange={e => setBatchSize(Number(e.target.value))}
                disabled={running}
              >
                {[10, 25, 50, 100, 200, 500].map(v => (
                  <option key={v} value={v}>{v.toLocaleString()}</option>
                ))}
              </select>
            </div>

            {/* Chunk size */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Chunk Size (words)</span>
              </label>
              <select
                className="select select-bordered select-sm w-28"
                value={chunkSize}
                onChange={e => setChunkSize(Number(e.target.value))}
                disabled={running}
              >
                {[150, 200, 300, 400].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>

            {/* Chunk overlap */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Chunk Overlap (words)</span>
              </label>
              <select
                className="select select-bordered select-sm w-24"
                value={chunkOverlap}
                onChange={e => setChunkOverlap(Number(e.target.value))}
                disabled={running}
              >
                {[0, 20, 40, 50].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>

            {/* Run button */}
            <button
              className="btn btn-primary btn-sm gap-2 self-end"
              onClick={runBenchmark}
              disabled={running}
            >
              {running ? (
                <>
                  <span className="loading loading-spinner loading-xs" />
                  Running {batchSize} questions…
                </>
              ) : (
                <>
                  <Play className="size-4" />
                  Run Batch
                </>
              )}
            </button>

            {/* Reset button */}
            <button
              className="btn btn-outline btn-sm gap-2 self-end"
              onClick={resetBenchmark}
              disabled={running || resetting}
              title="Discard saved progress for this method and start from question 0"
            >
              {resetting ? (
                <span className="loading loading-spinner loading-xs" />
              ) : (
                <RotateCcw className="size-4" />
              )}
              Reset
            </button>
          </div>

          {resumeIndex > 0 && (
            <div className="alert alert-info text-xs py-2">
              ℹ️  Resuming from question {resumeIndex.toLocaleString()}. Running another batch
              continues from here — use Reset to start this method over from question 0.
            </div>
          )}

          <div className="alert alert-warning text-xs py-2">
            ⚠️  Each question rebuilds a small index from scratch (ingest → retrieve → delete),
            so this is slower than the Groundedness benchmark. Start with 10–25 questions to
            confirm it runs, then increase the batch size.
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-error text-sm">
          <XCircle className="size-5" />
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {display && display.n_scored > 0 && (
        <div className="flex flex-col gap-4">
          <SummaryCard result={display} />

          {display.checkpoints.length > 0 && (
            <div className="card bg-base-100 shadow">
              <div className="card-body gap-3">
                <h3 className="card-title text-base">Progress Across the Run</h3>
                <CheckpointTable checkpoints={display.checkpoints} topK={display.top_k} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PrecisionBenchmark;
