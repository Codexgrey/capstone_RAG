import React from "react";
import { MessageSquare, User } from "lucide-react";

interface Message {
  id?:               string;
  role:              "user" | "assistant";
  content:           string;
  created_at?:       string;
  retrieval_method?: string | null;
}

interface ResponseCardProps {
  response:  string;
  loading?:  boolean;
  messages?: Message[];
}

/* ── Lightweight inline markdown renderer ───────────────────────────────────
   Handles the subset the LLM actually emits:
   - **bold**  |  *italic*  |  `inline code`
   - # / ## / ### headings
   - bullet lists  (- / * ...)
   - numbered lists  (1. ...)
   - blank lines → paragraph spacing
   No external dependency required.
─────────────────────────────────────────────────────────────────────────── */
function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let key = 0;

  function inlineFormat(line: string): React.ReactNode[] {
    const parts: React.ReactNode[] = [];
    const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`)/g;
    let last = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(line)) !== null) {
      if (match.index > last) parts.push(line.slice(last, match.index));
      if (match[0].startsWith("**"))
        parts.push(<strong key={key++}>{match[2]}</strong>);
      else if (match[0].startsWith("*"))
        parts.push(<em key={key++}>{match[3]}</em>);
      else
        parts.push(
          <code key={key++} className="bg-base-300 px-1 rounded text-xs font-mono">
            {match[4]}
          </code>
        );
      last = match.index + match[0].length;
    }
    if (last < line.length) parts.push(line.slice(last));
    return parts;
  }

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Headings
    const h1 = line.match(/^#\s+(.*)/);
    const h2 = line.match(/^##\s+(.*)/);
    const h3 = line.match(/^###\s+(.*)/);
    if (h3) { nodes.push(<h3 key={key++} className="text-sm font-bold mt-2 mb-0.5">{inlineFormat(h3[1])}</h3>); i++; continue; }
    if (h2) { nodes.push(<h2 key={key++} className="text-base font-bold mt-3 mb-1">{inlineFormat(h2[1])}</h2>); i++; continue; }
    if (h1) { nodes.push(<h1 key={key++} className="text-lg font-bold mt-3 mb-1">{inlineFormat(h1[1])}</h1>); i++; continue; }

    // Bullet list block
    if (/^[-*]\s+/.test(line)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(<li key={key++} className="ml-4 list-disc">{inlineFormat(lines[i].replace(/^[-*]\s+/, ""))}</li>);
        i++;
      }
      nodes.push(<ul key={key++} className="my-1 space-y-0.5">{items}</ul>);
      continue;
    }

    // Numbered list block
    if (/^\d+\.\s+/.test(line)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(<li key={key++} className="ml-4 list-decimal">{inlineFormat(lines[i].replace(/^\d+\.\s+/, ""))}</li>);
        i++;
      }
      nodes.push(<ol key={key++} className="my-1 space-y-0.5">{items}</ol>);
      continue;
    }

    // Blank line spacer
    if (line.trim() === "") { nodes.push(<div key={key++} className="h-2" />); i++; continue; }

    // Plain paragraph
    nodes.push(<p key={key++} className="leading-relaxed">{inlineFormat(line)}</p>);
    i++;
  }
  return nodes;
}

/* ── Section-aware structured answer renderer ───────────────────────────────
   Detects Answer: / Evidence Used: / Citations: headers and styles them.
─────────────────────────────────────────────────────────────────────────── */
function StructuredAnswer({ content }: { content: string }) {
  const sectionRe = /^(Answer:|Evidence Used:|Citations:)/im;

  if (!sectionRe.test(content)) {
    return <div className="space-y-1 text-sm">{renderMarkdown(content)}</div>;
  }

  const tokens = content.split(/(Answer:|Evidence Used:|Citations:)/i).filter(Boolean);
  const sections: { label: string; body: string }[] = [];
  for (let t = 0; t < tokens.length; t++) {
    if (/^(Answer:|Evidence Used:|Citations:)$/i.test(tokens[t].trim())) {
      sections.push({ label: tokens[t].trim(), body: (tokens[t + 1] || "").trim() });
      t++;
    }
  }

  return (
    <div className="space-y-2">
      {sections.map((s, i) => (
        <div key={i}>
          <p className="text-xs font-bold uppercase tracking-widest text-primary mb-1 mt-3 first:mt-0">
            {s.label}
          </p>
          <div className="text-base-content/80 text-sm space-y-1">
            {renderMarkdown(s.body)}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Main component ─────────────────────────────────────────────────────── */
const ResponseCard: React.FC<ResponseCardProps> = ({
  response,
  loading = false,
  messages = [],
}) => {

  /* Full session thread view */
  if (messages.length > 0) {
    return (
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body gap-4">
          <h3 className="card-title text-base-content flex items-center gap-2">
            <MessageSquare className="size-5 text-primary" />
            Conversation
          </h3>
          <div className="flex flex-col gap-3 max-h-96 overflow-y-auto pr-1">
            {messages.map((m, i) => (
              <div key={m.id || i} className={`flex gap-2 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                {m.role === "assistant" && (
                  <div className="avatar placeholder shrink-0">
                    <div className="bg-primary text-primary-content rounded-full w-7 h-7 flex items-center justify-center">
                      <MessageSquare className="size-3" />
                    </div>
                  </div>
                )}
                <div className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm
                  ${m.role === "user"
                    ? "bg-primary text-primary-content rounded-tr-none"
                    : "bg-base-200 text-base-content rounded-tl-none"}`}
                >
                  {m.role === "assistant"
                    ? <StructuredAnswer content={m.content} />
                    : <p className="leading-relaxed">{m.content}</p>
                  }
                  {m.retrieval_method && m.role === "assistant" && (
                    <span className="text-xs opacity-50 mt-1 block">via {m.retrieval_method}</span>
                  )}
                </div>
                {m.role === "user" && (
                  <div className="avatar placeholder shrink-0">
                    <div className="bg-base-300 text-base-content rounded-full w-7 h-7 flex items-center justify-center">
                      <User className="size-3" />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* Default — current answer */
  return (
    <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200 min-h-48">
      <div className="card-body">
        <h3 className="card-title text-base-content flex items-center gap-2">
          <MessageSquare className="size-5 text-primary" />
          Response
        </h3>
        {loading ? (
          <div className="flex flex-col items-center justify-center py-10 gap-3">
            <span className="loading loading-dots loading-lg text-primary" />
            <p className="text-base-content/50 text-sm">Searching documents and generating answer...</p>
          </div>
        ) : response ? (
          <div className="mt-2">
            <StructuredAnswer content={response} />
          </div>
        ) : (
          <p className="text-base-content/40 text-sm italic mt-2">
            Ask a question to see the RAG system response.
          </p>
        )}
      </div>
    </div>
  );
};

export default ResponseCard;
