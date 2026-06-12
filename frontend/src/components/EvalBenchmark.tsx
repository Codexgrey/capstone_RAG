/**
 * EvalBenchmark.tsx
 * =================
 * TriviaQA batch evaluation panel — 5,000-question bank.
 *
 * Lets the user configure and run the TriviaQA benchmark against any of the
 * three retrieval methods, with filtering by domain, answer type, and batch size.
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  FlaskConical, Play, ChevronDown, ChevronUp,
  CheckCircle, XCircle, AlertCircle, BarChart2, Filter,
} from "lucide-react";

// Re-use the API base URL already defined in queryService
const API_BASE = (import.meta as { env: { VITE_API_URL?: string } }).env?.VITE_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BankStats {
  total:       number;
  domains:     Record<string, number>;
  types:       Record<string, number>;
  avg_aliases: number;
}

interface QuestionResult {
  question_id:  string;
  question:     string;
  domain:       string;
  answer_type:  string;
  predicted:    string;
  ground_truth: string;
  aliases:      string[];
  exact_match:  number;
  f1:           number;
}

interface BatchResult {
  method:   string;
  total:    number;
  answered: number;
  avg_em:   number;
  avg_f1:   number;
  results:  QuestionResult[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const METHOD_LABELS: Record<string, string> = {
  vector:  "Vector (FAISS)",
  keyword: "Keyword (BM25)",
  hybrid:  "Hybrid (FAISS+BM25)",
};

const DOMAIN_COLORS: Record<string, string> = {
  Wikipedia: "badge-primary",
  Web:       "badge-secondary",
};

const TYPE_LABELS: Record<string, string> = {
  WikipediaEntity: "Entity",
  Numerical:       "Number",
  FreeForm:        "Free-form",
  Range:           "Range",
};

function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function ScoreChip({ value }: { value: number }) {
  const p   = Math.round(value * 100);
  const cls = p === 100 ? "badge-success" : p >= 50 ? "badge-warning" : "badge-error";
  return <span className={`badge badge-sm ${cls} font-mono`}>{p}%</span>;
}

function EMIcon({ em }: { em: number }) {
  if (em === 1) return <CheckCircle className="size-4 text-success shrink-0" />;
  if (em > 0)   return <AlertCircle className="size-4 text-warning shrink-0" />;
  return              <XCircle      className="size-4 text-error   shrink-0" />;
}

// ---------------------------------------------------------------------------
// Bank stats banner
// ---------------------------------------------------------------------------

function StatsBanner({ stats }: { stats: BankStats }) {
  return (
    <div className="stats stats-horizontal shadow w-full bg-base-200 text-sm">
      <div className="stat py-3 px-4">
        <div className="stat-title text-xs">Total Questions</div>
        <div className="stat-value text-2xl">{stats.total.toLocaleString()}</div>
        <div className="stat-desc">TriviaQA dev split</div>
      </div>
      {Object.entries(stats.domains).map(([d, n]) => (
        <div key={d} className="stat py-3 px-4">
          <div className="stat-title text-xs">{d}</div>
          <div className="stat-value text-2xl">{n.toLocaleString()}</div>
          <div className="stat-desc">questions</div>
        </div>
      ))}
      <div className="stat py-3 px-4">
        <div className="stat-title text-xs">Avg Aliases / Q</div>
        <div className="stat-value text-2xl">{stats.avg_aliases}</div>
        <div className="stat-desc">answer variants</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Summary cards after a run
// ---------------------------------------------------------------------------

function SummaryCard({ result }: { result: BatchResult }) {
  return (
    <div className="stats stats-horizontal shadow w-full bg-base-200">
      <div className="stat">
        <div className="stat-title">Method</div>
        <div className="stat-value text-lg">{METHOD_LABELS[result.method] ?? result.method}</div>
      </div>
      <div className="stat">
        <div className="stat-title">Avg EM</div>
        <div className="stat-value text-success">{pct(result.avg_em)}</div>
        <div className="stat-desc">Exact Match</div>
      </div>
      <div className="stat">
        <div className="stat-title">Avg F1</div>
        <div className="stat-value text-info">{pct(result.avg_f1)}</div>
        <div className="stat-desc">Token F1</div>
      </div>
      <div className="stat">
        <div className="stat-title">Questions Run</div>
        <div className="stat-value">{result.answered} / {result.total}</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Expandable question row
// ---------------------------------------------------------------------------

function QuestionRow({ item }: { item: QuestionResult }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-base-300 rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-base-200 transition-colors text-left"
        onClick={() => setOpen(o => !o)}
      >
        <EMIcon em={item.exact_match} />
        <span className="badge badge-ghost badge-xs font-mono shrink-0">{item.question_id}</span>
        <span className={`badge badge-xs ${DOMAIN_COLORS[item.domain] ?? "badge-ghost"} shrink-0`}>
          {item.domain}
        </span>
        <span className="badge badge-xs badge-outline shrink-0">
          {TYPE_LABELS[item.answer_type] ?? item.answer_type}
        </span>
        <span className="flex-1 text-sm text-base-content line-clamp-1 min-w-0">
          {item.question}
        </span>
        <ScoreChip value={item.exact_match} />
        <ScoreChip value={item.f1} />
        {open
          ? <ChevronUp   className="size-4 shrink-0 text-base-content/40" />
          : <ChevronDown className="size-4 shrink-0 text-base-content/40" />}
      </button>

      {open && (
        <div className="px-4 pb-4 pt-2 flex flex-col gap-2 bg-base-50 text-sm border-t border-base-300">
          <div>
            <p className="text-base-content/40 text-xs mb-0.5">Question</p>
            <p>{item.question}</p>
          </div>
          <div>
            <p className="text-base-content/40 text-xs mb-0.5">RAG Answer</p>
            <p className="bg-base-200 rounded p-2 text-xs font-mono whitespace-pre-wrap">
              {item.predicted || <em className="opacity-40">— no answer generated —</em>}
            </p>
          </div>
          <div className="flex gap-6 flex-wrap">
            <div>
              <p className="text-base-content/40 text-xs mb-0.5">Ground Truth</p>
              <p className="font-semibold">{item.ground_truth}</p>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-base-content/40 text-xs mb-0.5">
                Valid Aliases ({item.aliases.length})
              </p>
              <p className="text-base-content/60 text-xs break-words">
                {item.aliases.slice(0, 8).join(", ")}
                {item.aliases.length > 8 && ` … +${item.aliases.length - 8} more`}
              </p>
            </div>
          </div>
          <div className="flex gap-4 mt-1 text-xs">
            <span>EM: <ScoreChip value={item.exact_match} /></span>
            <span>F1: <ScoreChip value={item.f1} /></span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const EvalBenchmark: React.FC = () => {
  const [stats,   setStats]   = useState<BankStats | null>(null);
  const [method,  setMethod]  = useState("vector");
  const [topK,    setTopK]    = useState(5);
  const [n,       setN]       = useState(50);
  const [domain,  setDomain]  = useState("");
  const [qtype,   setQtype]   = useState("");
  const [running, setRunning] = useState(false);
  const [result,  setResult]  = useState<BatchResult | null>(null);
  const [error,   setError]   = useState<string | null>(null);
  const [filter,  setFilter]  = useState<"all" | "correct" | "partial" | "wrong">("all");

  const token = () => localStorage.getItem("token") ?? "";

  // Load bank stats on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/evaluate/triviaqa/stats`, {
      headers: { Authorization: `Bearer ${token()}` },
    })
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setStats(d))
      .catch(() => null);
  }, []);

  const runBenchmark = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);

    const body: Record<string, unknown> = {
      retrieval_method: method,
      top_k: topK,
      n,
    };
    if (domain) body.domain = domain;
    if (qtype)  body.qtype  = qtype;

    try {
      const res = await fetch(`${API_BASE}/api/evaluate/triviaqa/run`, {
        method:  "POST",
        headers: {
          "Content-Type":  "application/json",
          Authorization:   `Bearer ${token()}`,
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: unknown) {
      if (e instanceof Error) {
        setError(e.message);
      } else {
        setError("Unknown error");
      }
    } finally {
      setRunning(false);
    }
  }, [method, topK, n, domain, qtype]);

  const filteredResults = result?.results.filter(r => {
    if (filter === "correct") return r.exact_match === 1;
    if (filter === "partial") return r.exact_match > 0 && r.exact_match < 1;
    if (filter === "wrong")   return r.exact_match === 0;
    return true;
  }) ?? [];

  const correctCount = result?.results.filter(r => r.exact_match === 1).length  ?? 0;
  const partialCount = result?.results.filter(r => r.exact_match > 0 && r.exact_match < 1).length ?? 0;
  const wrongCount   = result?.results.filter(r => r.exact_match === 0).length ?? 0;

  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto p-4">

      {/* Header */}
      <div className="flex items-center gap-3">
        <FlaskConical className="size-6 text-primary" />
        <div>
          <h2 className="text-xl font-bold">TriviaQA Benchmark</h2>
          <p className="text-base-content/50 text-sm">
            Official EM &amp; F1 evaluation — 5,000-question bank (Joshi et al., ACL 2017)
          </p>
        </div>
        {stats && (
          <span className="badge badge-primary ml-auto">
            {stats.total.toLocaleString()} questions loaded
          </span>
        )}
      </div>

      {/* Bank stats */}
      {stats && <StatsBanner stats={stats} />}

      {/* Info */}
      <div className="alert text-sm bg-success text-white border border-success/20 shadow-sm">
        <BarChart2 className="size-5 shrink-0" />
        <div>
          <strong>How this works:</strong> The RAG pipeline answers each TriviaQA question
          using the ingested documents. Answers are scored against all valid aliases using
          the official TriviaQA Exact Match and token-level F1 functions; the same metrics
          reported in the paper. Questions come from <em>wikipedia-dev</em> (4,000) and
          <em> web-dev</em> (1,000 unique).
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

            {/* N questions */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Questions to Run</span>
              </label>
              <select
                className="select select-bordered select-sm w-28"
                value={n}
                onChange={e => setN(Number(e.target.value))}
                disabled={running}
              >
                {[10, 15, 20, 25, 50, 100, 250, 500, 1000, 5000].map(v => (
                  <option key={v} value={v}>{v.toLocaleString()}</option>
                ))}
              </select>
            </div>

            {/* Domain filter */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Domain</span>
              </label>
              <select
                className="select select-bordered select-sm w-36"
                value={domain}
                onChange={e => setDomain(e.target.value)}
                disabled={running}
              >
                <option value="">All domains</option>
                <option value="Wikipedia">Wikipedia</option>
                <option value="Web">Web</option>
              </select>
            </div>

            {/* Answer type filter */}
            <div className="form-control">
              <label className="label pb-1">
                <span className="label-text text-xs">Answer Type</span>
              </label>
              <select
                className="select select-bordered select-sm w-36"
                value={qtype}
                onChange={e => setQtype(e.target.value)}
                disabled={running}
              >
                <option value="">All types</option>
                <option value="WikipediaEntity">Entity</option>
                <option value="Numerical">Numerical</option>
                <option value="FreeForm">Free-form</option>
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
                  Running {n} questions…
                </>
              ) : (
                <>
                  <Play className="size-4" />
                  Run Benchmark
                </>
              )}
            </button>
          </div>

          {n > 100 && (
            <div className="alert alert-warning text-xs py-2">
              ⚠️  Batches above 100 questions may take several minutes due to LLM API rate limits.
              Start with 50–100 for interactive testing.
            </div>
          )}
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
      {result && (
        <div className="flex flex-col gap-4">
          <SummaryCard result={result} />

          <div className="card bg-base-100 shadow">
            <div className="card-body gap-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h3 className="card-title text-base">Per-Question Results</h3>

                {/* Filter tabs */}
                <div className="join">
                  {([
                    ["all",     `All (${result.total})`],
                    ["correct", `✓ ${correctCount}`],
                    ["partial", `~ ${partialCount}`],
                    ["wrong",   `✗ ${wrongCount}`],
                  ] as const).map(([f, label]) => (
                    <button
                      key={f}
                      className={`join-item btn btn-xs ${filter === f ? "btn-primary" : "btn-ghost"}`}
                      onClick={() => setFilter(f)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Column hints */}
              <div className="flex items-center gap-2 px-3 text-xs text-base-content/30 uppercase tracking-wide">
                <span className="w-4" />
                <span className="w-24 shrink-0">ID</span>
                <span className="flex-1">Question</span>
                <span className="w-10 text-center">EM</span>
                <span className="w-10 text-center">F1</span>
                <span className="w-4" />
              </div>

              <div className="flex flex-col gap-1.5">
                {filteredResults.map(item => (
                  <QuestionRow key={item.question_id} item={item} />
                ))}
                {filteredResults.length === 0 && (
                  <p className="text-base-content/40 text-sm italic text-center py-4">
                    No questions match this filter.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EvalBenchmark;
