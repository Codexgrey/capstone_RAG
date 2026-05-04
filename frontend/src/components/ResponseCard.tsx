import React from "react";
import { MessageSquare, User } from "lucide-react";

interface Message {
  id?:              string;
  role:             "user" | "assistant";
  content:          string;
  created_at?:      string;
  retrieval_method?: string | null;
}

interface ResponseCardProps {
  response:  string;
  loading?:  boolean;
  messages?: Message[];   // full conversation thread from history
}

const ResponseCard: React.FC<ResponseCardProps> = ({
  response,
  loading = false,
  messages = [],
}) => {

  /* ── If we have a loaded session — show full thread ── */
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
              <div
                key={m.id || i}
                className={`flex gap-2 ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {m.role === "assistant" && (
                  <div className="avatar placeholder shrink-0">
                    <div className="bg-primary text-primary-content rounded-full w-7 h-7 flex items-center justify-center">
                      <MessageSquare className="size-3" />
                    </div>
                  </div>
                )}

                <div
                  className={`
                    max-w-[80%] rounded-2xl px-4 py-2 text-sm leading-relaxed
                    ${m.role === "user"
                      ? "bg-primary text-primary-content rounded-tr-none"
                      : "bg-base-200 text-base-content rounded-tl-none"}
                  `}
                >
                  <p className="whitespace-pre-wrap">{m.content}</p>
                  {m.retrieval_method && m.role === "assistant" && (
                    <span className="text-xs opacity-50 mt-1 block">
                      via {m.retrieval_method}
                    </span>
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

  /* ── Default — show current answer ── */
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
            <p className="text-base-content/50 text-sm">
              Searching documents and generating answer...
            </p>
          </div>
        ) : (
          <p className="text-base-content/80 leading-relaxed mt-2 whitespace-pre-wrap">
            {response || "Ask a question to see the RAG system response."}
          </p>
        )}
      </div>
    </div>
  );
};

export default ResponseCard;