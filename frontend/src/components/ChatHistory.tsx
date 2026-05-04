import React, { useEffect, useState } from "react";
import { MessageSquare, ChevronRight, Clock, Trash2 } from "lucide-react";
import { getSessions, getSessionMessages } from "../services/queryService";

interface ChatSession {
  session_id: string;
  title:      string;
  created_at: string;
  updated_at: string;
}

interface ChatMessage {
  id:               string;
  role:             "user" | "assistant";
  content:          string;
  created_at:       string;
  retrieval_method: string | null;
}

interface ChatHistoryProps {
  currentSessionId: string | null;
  onLoadSession:    (sessionId: string, messages: ChatMessage[]) => void;
  onNewChat:        () => void;
  refreshTrigger:   number;   // increment this from App.tsx to force refresh
}

const ChatHistory: React.FC<ChatHistoryProps> = ({
  currentSessionId,
  onLoadSession,
  onNewChat,
  refreshTrigger,
}) => {
  const [sessions, setSessions]   = useState<ChatSession[]>([]);
  const [loading, setLoading]     = useState(false);
  const [expanded, setExpanded]   = useState(true);

  /* ── Load sessions whenever a new message is sent ── */
  useEffect(() => {
    fetchSessions();
  }, [refreshTrigger]);

  const fetchSessions = async () => {
    setLoading(true);
    try {
      const data = await getSessions();
      setSessions(data.sessions || []);
    } catch {
      // not logged in yet or backend not ready
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSession = async (sessionId: string) => {
    try {
      const data = await getSessionMessages(sessionId);
      onLoadSession(sessionId, data.messages || []);
    } catch {
      // session not found
    }
  };

  const formatTime = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
           " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
      <div className="card-body p-4 gap-3">

        {/* Header */}
        <div className="flex items-center justify-between">
          <button
            className="flex items-center gap-2 text-sm font-semibold text-base-content/70 uppercase tracking-widest hover:text-primary transition-colors"
            onClick={() => setExpanded(!expanded)}
          >
            <MessageSquare className="size-4" />
            Chat History
            <ChevronRight
              className={`size-3 transition-transform ${expanded ? "rotate-90" : ""}`}
            />
          </button>

          <button
            className="btn btn-primary btn-xs gap-1"
            onClick={onNewChat}
            title="Start new chat"
          >
            + New
          </button>
        </div>

        {/* Session list */}
        {expanded && (
          <div className="flex flex-col gap-1 max-h-64 overflow-y-auto">
            {loading ? (
              <div className="flex justify-center py-4">
                <span className="loading loading-dots loading-sm text-primary" />
              </div>
            ) : sessions.length === 0 ? (
              <p className="text-base-content/40 text-xs italic py-2 text-center">
                No past conversations yet
              </p>
            ) : (
              sessions.map((s) => {
                const isActive = s.session_id === currentSessionId;
                return (
                  <button
                    key={s.session_id}
                    onClick={() => handleLoadSession(s.session_id)}
                    className={`
                      flex items-start gap-2 w-full text-left rounded-lg px-3 py-2
                      transition-all duration-150 hover:bg-base-200
                      ${isActive ? "bg-base-200 border border-primary/30" : ""}
                    `}
                  >
                    <MessageSquare className={`size-3 mt-1 shrink-0 ${isActive ? "text-primary" : "text-base-content/40"}`} />
                    <div className="flex flex-col gap-0.5 min-w-0">
                      <span className={`text-xs truncate ${isActive ? "text-primary font-medium" : "text-base-content/70"}`}>
                        {s.title || "Chat"}
                      </span>
                      <span className="text-xs text-base-content/40 flex items-center gap-1">
                        <Clock className="size-2" />
                        {formatTime(s.updated_at)}
                      </span>
                    </div>
                    {isActive && (
                      <span className="badge badge-primary badge-xs ml-auto mt-1 shrink-0">
                        active
                      </span>
                    )}
                  </button>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatHistory;