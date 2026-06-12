import React, { useEffect, useState, useRef } from "react";
import { MessageSquare, ChevronRight, Clock, Trash2, MoreVertical, Pencil, Check, X } from "lucide-react";
import { getSessions, getSessionMessages, renameSession, deleteSession, deleteAllSessions } from "../services/queryService";


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
  const [menuOpenId, setMenuOpenId]     = useState<string | null>(null);
  const [menuPos, setMenuPos]           = useState<{ top: number; right: number } | null>(null);
  const [renamingId, setRenamingId]     = useState<string | null>(null);
  const [renameValue, setRenameValue]   = useState("");
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  /* ── Load sessions whenever a new message is sent ── */
  useEffect(() => {
    fetchSessions();
  }, [refreshTrigger]);

  /* ── Close kebab menu on outside click ── */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
        setMenuPos(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleKebabClick = (e: React.MouseEvent<HTMLButtonElement>, sessionId: string) => {
    e.stopPropagation();
    if (menuOpenId === sessionId) {
      setMenuOpenId(null);
      setMenuPos(null);
      return;
    }
    // Calculate fixed position from button's bounding rect
    const rect = e.currentTarget.getBoundingClientRect();
    setMenuPos({
      top:   rect.bottom + 4,
      right: window.innerWidth - rect.right,
    });
    setMenuOpenId(sessionId);
  };

  const handleRenameCommit = async (sessionId: string) => {
    const trimmed = renameValue.trim();
    if (trimmed) {
      try {
        await renameSession(sessionId, trimmed);
        setSessions((prev) =>
          prev.map((s) => s.session_id === sessionId ? { ...s, title: trimmed } : s)
        );
      } catch { /* silent — title stays as-is */ }
    }
    setRenamingId(null);
    setMenuOpenId(null);
    setMenuPos(null);
  };

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
      if (sessionId === currentSessionId) onNewChat();
    } catch { /* silent */ }
    setMenuOpenId(null);
    setMenuPos(null);
  };

  const handleClearAll = async () => {
    if (!confirmClearAll) {
      setConfirmClearAll(true);
      return;
    }
    try {
      await deleteAllSessions();
      setSessions([]);
      onNewChat();
    } catch { /* silent */ }
    setConfirmClearAll(false);
  };

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
            Chats
            <ChevronRight
              className={`size-3 transition-transform ${expanded ? "rotate-90" : ""}`}
            />
          </button>

          <div className="flex items-center gap-1">
            {sessions.length > 0 && (
              <button
                className={`btn btn-xs gap-1 ${confirmClearAll ? "btn-error" : "btn-ghost text-error"}`}
                onClick={handleClearAll}
                onBlur={() => setConfirmClearAll(false)}
                title="Delete all chat history"
              >
                <Trash2 className="size-3" />
                {confirmClearAll ? "Confirm?" : "Clear All"}
              </button>
            )}
            <button
              className="btn btn-primary btn-xs gap-1"
              onClick={onNewChat}
              title="Start new chat"
            >
              + New
            </button>
          </div>
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
              <div className="flex flex-col gap-1">
              {sessions.map((s) => {
                const isActive   = s.session_id === currentSessionId;
                //const menuOpen   = menuOpenId === s.session_id;
                const isRenaming = renamingId === s.session_id;

                return (
                  <div
                    key={s.session_id}
                    className={`
                      group relative flex items-center gap-2 w-full rounded-lg px-3 py-2
                      transition-all duration-150 hover:bg-base-200
                      ${isActive ? "bg-base-200 border border-primary/30" : ""}
                    `}
                  >
                    <MessageSquare className={`size-3 shrink-0 ${isActive ? "text-primary" : "text-base-content/40"}`} />

                    {/* Title or rename input */}
                    {isRenaming ? (
                      <div className="flex items-center gap-1 flex-1 min-w-0">
                        <input
                          autoFocus
                          className="input input-xs input-bordered flex-1 min-w-0"
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter")  handleRenameCommit(s.session_id);
                            if (e.key === "Escape") { setRenamingId(null); setMenuOpenId(null); setMenuPos(null); }
                          }}
                        />
                        <button className="btn btn-ghost btn-xs px-1" onClick={() => handleRenameCommit(s.session_id)}>
                          <Check className="size-3 text-success" />
                        </button>
                        <button className="btn btn-ghost btn-xs px-1" onClick={() => { setRenamingId(null); setMenuOpenId(null); setMenuPos(null); }}>
                          <X className="size-3 text-error" />
                        </button>
                      </div>
                    ) : (
                      <button
                        className="flex flex-col gap-0.5 min-w-0 flex-1 text-left"
                        onClick={() => handleLoadSession(s.session_id)}
                      >
                        <span className={`text-xs truncate ${isActive ? "text-primary font-medium" : "text-base-content/70"}`}>
                          {s.title || "Chat"}
                        </span>
                        <span className="text-xs text-base-content/40 flex items-center gap-1">
                          <Clock className="size-2" />
                          {formatTime(s.updated_at)}
                        </span>
                      </button>
                    )}

                    {isActive && !isRenaming && (
                      <span className="badge badge-primary badge-xs shrink-0">active</span>
                    )}

                    {/* Kebab menu button */}
                    {!isRenaming && (
                      <button
                        className="btn btn-ghost btn-xs px-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                        onClick={(e) => handleKebabClick(e, s.session_id)}
                        title="Options"
                      >
                        <MoreVertical className="size-3" />
                      </button>
                    )}
                  </div>
                );
              })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Fixed-position dropdown — renders outside scrollable container */}
      {menuOpenId && menuPos && (
        <div
          ref={menuRef}
          style={{ position: "fixed", top: menuPos.top, right: menuPos.right, zIndex: 9999 }}
          className="w-32 bg-base-100 border border-base-content/10 rounded-lg shadow-xl py-1"
        >
          <button
            className="w-full text-left px-3 py-2 text-xs text-base-content hover:bg-base-200 flex items-center gap-2"
            onClick={() => {
              const s = sessions.find(s => s.session_id === menuOpenId);
              if (s) { setRenamingId(s.session_id); setRenameValue(s.title || ""); }
              setMenuOpenId(null);
              setMenuPos(null);
            }}
          >
            <Pencil className="size-3" /> Rename
          </button>
          <button
            className="w-full text-left px-3 py-2 text-xs text-error hover:bg-base-200 flex items-center gap-2"
            onClick={() => handleDelete(menuOpenId)}
          >
            <Trash2 className="size-3" /> Delete
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatHistory;
