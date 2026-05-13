import React, { useState } from "react";
import { Send, Cpu } from "lucide-react";

interface ChatBoxProps {
  onSend:   (query: string) => void;
  loading?: boolean;
  method:   string;
  setMethod: (m: string) => void;
}

const METHODS = [
  { value: "vector",  label: "Vector",  desc: "Semantic similarity (FAISS)" },
  { value: "keyword", label: "Keyword", desc: "BM25 keyword search" },
  { value: "hybrid", label: "Hybrid",  desc: "FAISS + BM25 + RRF fusion" },
];

const ChatBox: React.FC<ChatBoxProps> = ({ onSend, loading = false, method, setMethod }) => {
  const [query, setQuery] = useState("");

  const handleSend = () => {
    if (!query.trim() || loading) return;
    onSend(query);
    setQuery("");
  };

  return (
    <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
      <div className="card-body gap-4">
        <h2 className="card-title text-base-content text-lg">Ask a Question</h2>

        {/* Retrieval method selector */}
        <div className="flex flex-wrap gap-2">
          {METHODS.map((m) => (
            <button
              key={m.value}
              className={`btn btn-sm gap-1 transition-all ${
                method === m.value
                  ? "btn-primary"
                  : "btn-ghost border border-base-content/20"
              }`}
              onClick={() => setMethod(m.value)}
              title={m.desc}
            >
              <Cpu className="size-3" />
              {m.label}
            </button>
          ))}
          <span className="text-xs text-base-content/40 self-center ml-1">
            {METHODS.find((m) => m.value === method)?.desc}
          </span>
        </div>

        <textarea
          className="textarea textarea-bordered w-full h-24 resize-none text-base leading-relaxed"
          placeholder="Ask something about your documents..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={loading}
        />

        <div className="card-actions justify-between items-center">
          <span className="text-xs text-base-content/40">
            Press Enter to send · Shift+Enter for new line
          </span>
          <button
            className="btn btn-primary gap-2"
            onClick={handleSend}
            disabled={!query.trim() || loading}
          >
            {loading ? (
              <>
                <span className="loading loading-spinner loading-xs" />
                Thinking...
              </>
            ) : (
              <>
                <Send className="size-4" />
                Send
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBox;
