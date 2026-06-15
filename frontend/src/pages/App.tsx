import { useState, useEffect } from "react";
import { Toaster, toast } from "react-hot-toast";

import Sidebar        from "../components/SideBar";
import ChatBox        from "../components/ChatBox";
import ResponseCard   from "../components/ResponseCard";
import SourcesPanel   from "../components/SourcesPanel";
import Header         from "../components/Header";
import ChatHistory    from "../components/ChatHistory";
import Admin              from "../components/admin";
import EvalBenchmark      from "../components/EvalBenchmark";       // ← Groundedness
import PrecisionBenchmark from "../components/PrecisionBenchmark";  // ← Precision

import { login, logout, register } from "../services/authService";
import { sendQuery }               from "../services/queryService";
import { Mail, Lock, User as UserIcon } from "lucide-react";

type Theme    = "forest" | "raglight";
type User     = { username: string; password: string; role: "admin"|"user"; };
type Citation = { chunk_id: string; source_name: string; page: number; section: string|null; };
type Latency  = { retrieval: number; llm: number; };
type Message  = { id?: string; role: "user"|"assistant"; content: string; created_at?: string; retrieval_method?: string|null; };
type EvalMetrics = {
  top_score?: number; avg_score?: number; source_coverage?: number;
  chunks_retrieved?: number; precision_at_k?: number; mrr?: number; source_diversity?: number;
  triviaqa_em?: number; triviaqa_f1?: number; triviaqa_qid?: string;   // ← TriviaQA fields
};

type MainLayoutProps = {
  response: string;
  setResponse: (value: string) => void;
  citations: Citation[];
  setCitations: (value: Citation[]) => void;
  latency: Latency;
  setLatency: (value: Latency) => void;
  onSend: (query: string) => Promise<void>;
  onLogout: () => void;
  loading: boolean;
  currentUser: User;
  sessionId: string | null;
  setSessionId: (value: string | null) => void;
  method: string;
  setMethod: (value: string) => void;
  menuOpen: boolean;
  setMenuOpen: (value: boolean) => void;
  showAdmin: boolean;
  setShowAdmin: (value: boolean) => void;
  theme: Theme;
  onThemeToggle: () => void;
  isAdmin: boolean;
  responseMethod: string;
  evalMetrics: EvalMetrics;
  setEvalMetrics: (value: EvalMetrics) => void;
};

/* ══════════════════════════════════════════════════════════
   MAIN LAYOUT
══════════════════════════════════════════════════════════ */
function MainLayout({ response, setResponse, citations, setCitations, latency, setLatency,
  onSend, onLogout, loading, currentUser, sessionId, setSessionId, method, setMethod,
  menuOpen, setMenuOpen, showAdmin, setShowAdmin, theme, onThemeToggle, isAdmin, responseMethod,
  evalMetrics, setEvalMetrics }: MainLayoutProps) {

  const [historyMessages, setHistoryMessages] = useState<Message[]>([]);
  const [refreshHistory,  setRefreshHistory]  = useState(0);
  const [showBenchmark,   setShowBenchmark]   = useState(false);  // ← Groundedness panel
  const [showPrecision,   setShowPrecision]   = useState(false);  // ← Precision panel

  const handleLoadSession = (sid: string, messages: Message[]) => {
    setSessionId(sid);
    setHistoryMessages(messages);
    setResponse("");
  };

  const handleNewChat = () => {
    setSessionId(null);
    setHistoryMessages([]);
    setResponse(""); setCitations([]); setLatency({ retrieval: 0, llm: 0 }); setEvalMetrics({});
    toast.success("New chat started");
  };

  const handleSend = async (query: string) => {
    setHistoryMessages([]);
    await onSend(query);
    setRefreshHistory((n: number) => n + 1);
  };

  return (
    <div className="min-h-screen bg-base-200">
      <Sidebar open={menuOpen} setOpen={setMenuOpen} setMethod={setMethod} />

      <Header
        setMenuOpen={setMenuOpen} showMenu={true}
        username={currentUser.username} onLogout={onLogout}
        onAdminToggle={isAdmin ? () => { setShowAdmin(!showAdmin); setShowBenchmark(false); setShowPrecision(false); } : undefined}
        showAdmin={showAdmin} theme={theme} onThemeToggle={onThemeToggle}
      />

      {/* ── TriviaQA Benchmark button row (visible to all logged-in users) ── */}
      <div className="max-w-7xl mx-auto px-6 pt-4 flex flex-col items-center gap-2">
        <span className="text-lg font-semibold  tracking-wide text-base-content/60">
          TriviaQA Benchmark
        </span>
        <div className="inline-flex rounded-full border border-base-300 p-1 gap-1">
          <button
            className={`btn btn-sm gap-2 rounded-full ${showBenchmark ? "btn-error" : "btn-outline"}`}
            onClick={() => { setShowBenchmark(b => !b); setShowPrecision(false); setShowAdmin(false); }}
            title="Toggle TriviaQA groundedness benchmark runner"
          >
            {/* flask icon via unicode — no extra import needed */}
            🧪 {showBenchmark ? "Close Groundedness" : "Groundedness"}
          </button>
          <button
            className={`btn btn-sm gap-2 rounded-full ${showPrecision ? "btn-error" : "btn-outline"}`}
            onClick={() => { setShowPrecision(p => !p); setShowBenchmark(false); setShowAdmin(false); }}
            title="Toggle TriviaQA precision/relevance benchmark runner"
          >
            🎯 {showPrecision ? "Close Precision" : "Precision"}
          </button>
        </div>
      </div>

      <main className="max-w-7xl mx-auto p-6 mt-2">
        {/* ── Admin panel ── */}
        {isAdmin && showAdmin ? (
          <Admin onLogout={onLogout} />

        /* ── TriviaQA groundedness benchmark panel ── */
        ) : showBenchmark ? (
          <EvalBenchmark />

        /* ── TriviaQA precision/relevance benchmark panel ── */
        ) : showPrecision ? (
          <PrecisionBenchmark />

        /* ── Normal chat layout ── */
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 min-w-0">

            {/* Chat history — 1 col */}
            <div className="lg:col-span-1 flex flex-col gap-4 min-w-0">
              <ChatHistory
                currentSessionId={sessionId}
                onLoadSession={handleLoadSession}
                onNewChat={handleNewChat}
                refreshTrigger={refreshHistory}
              />
            </div>

            {/* Response + Chat input — 2 cols */}
            <div className="lg:col-span-2 flex flex-col gap-6 min-w-0">
              <ResponseCard response={response} loading={loading} messages={historyMessages} />
              <ChatBox
                onSend={handleSend}
                loading={loading}
                method={method}
                setMethod={setMethod}
              />
            </div>

            {/* Sources — 1 col */}
            <div className="lg:col-span-1 min-w-0">
              <SourcesPanel
                citations={citations}
                latency_ms={latency.retrieval}
                retrieval_method={responseMethod}
                evaluation_metrics={evalMetrics}
              />
            </div>

          </div>
        )}
      </main>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   APP
══════════════════════════════════════════════════════════ */
export default function App() {
  // ── Theme ──────────────────────────────────────────────
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem("rag-theme") as Theme) || "raglight";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("rag-theme", theme);
  }, [theme]);

  const toggleTheme = () =>
    setTheme(t => t === "raglight" ? "forest" : "raglight");

  // ── App state ───────────────────────────────────────────
  const [menuOpen,   setMenuOpen]   = useState(false);
  const [response,   setResponse]   = useState("");
  const [citations,  setCitations]  = useState<Citation[]>([]);
  const [sessionId,  setSessionId]  = useState<string|null>(null);
  const [latency,    setLatency]    = useState<Latency>({ retrieval: 0, llm: 0 });
  const [method,     setMethod]     = useState("vector");
  const [currentUser,setCurrentUser]= useState<User|null>(null);
  const [showAdmin,  setShowAdmin]  = useState(false);
  const [loading,    setLoading]    = useState(false);
  const [responseMethod, setResponseMethod] = useState<string>("vector");
  const [evalMetrics,    setEvalMetrics]    = useState<EvalMetrics>({});

  // ── Auth form ───────────────────────────────────────────
  const [authMode,  setAuthMode]  = useState<"login"|"register">("login");
  const [email,     setEmail]     = useState("");
  const [password,  setPassword]  = useState("");
  const [username,  setUsername]  = useState("");

  // ── Init ────────────────────────────────────────────────
  useEffect(() => {
    const saved = localStorage.getItem("currentUser");
    if (saved) setCurrentUser(JSON.parse(saved));
  }, []);

  // ── Login ───────────────────────────────────────────────
  const handleLogin = async () => {
    if (!email || !password) { toast.error("Please enter your email and password"); return; }
    try {
      const result  = await login(email, password);
      const isAdmin = ["admin@admin.com"].includes(email.toLowerCase());
      const user: User = { username: result.user.username, password: "", role: isAdmin ? "admin" : "user" };
      localStorage.setItem("currentUser", JSON.stringify(user));
      setCurrentUser(user);
      toast.success(`Welcome back, ${result.user.username}!`);
    } catch { toast.error("Incorrect email or password"); }
  };

  // ── Register ────────────────────────────────────────────
  const handleRegister = async () => {
    if (!username || !email || !password) { toast.error("All fields are required"); return; }
    if (password.length < 8) { toast.error("Password must be at least 8 characters"); return; }
    try {
      await register(username, email, password);
      toast.success("Account created! You can now log in.");
      setAuthMode("login"); setUsername(""); setPassword("");
    } catch (err) {
      const errorMessage = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Registration failed";
      toast.error(errorMessage);
    }
  };

  // ── Logout ──────────────────────────────────────────────
  const handleLogout = () => {
    logout(); localStorage.removeItem("currentUser");
    setCurrentUser(null); setShowAdmin(false);
    setSessionId(null); setResponse(""); setCitations([]);
    toast.success("Logged out");
  };

  // ── Query ───────────────────────────────────────────────
  const handleSend = async (query: string) => {
    setLoading(true);
    try {
      const result = await sendQuery(query, sessionId || undefined, method);
      setResponse(result.answer);
      setSessionId(result.session_id);
      setCitations(result.citations || []);
      setLatency({ retrieval: result.latency_ms, llm: 0 });
      setResponseMethod(result.retrieval_method || method);
      setEvalMetrics(result.evaluation_metrics || {});
    } catch {
      setResponse("Error contacting backend.");
      toast.error("Failed to get a response — is the backend running?");
    } finally { setLoading(false); }
  };

  // ── AUTH PAGE ───────────────────────────────────────────
  if (!currentUser) {
    return (
      <>
        <Toaster position="top-right" />
        <div className="min-h-screen flex flex-col items-center justify-center bg-base-200 gap-6 p-4">

          {/* Theme toggle on login page */}
          <div className="absolute top-4 right-4">
            <button className="btn btn-ghost btn-circle btn-sm" onClick={toggleTheme}
              title={theme === "forest" ? "Switch to light mode" : "Switch to dark mode"}>
              {theme === "forest"
                ? <span className="text-yellow-400 text-lg">☀️</span>
                : <span className="text-lg">🌙</span>}
            </button>
          </div>

          {/* Logo + title */}
          <div className="flex flex-col items-center gap-2">
            <img src="/logo.jpg" alt="Logo" className="w-14 h-14 rounded-xl object-contain"
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
            <h1 className="text-4xl font-bold text-primary font-mono tracking-tight">RAG System</h1>
            <p className="text-base-content/50 text-sm">Retrieval Augmented Generation</p>
          </div>

          {/* Auth card */}
          <div className="card bg-base-100 border-t-4 border-[#00FF9D] w-full max-w-md hover:shadow-lg transition-all duration-200">
            <div className="card-body gap-4">

              {/* Tab switcher */}
              <div className="tabs tabs-boxed">
                <button className={`tab flex-1 ${authMode==="login"?"tab-active":""}`}
                  onClick={() => setAuthMode("login")}>Login</button>
                <button className={`tab flex-1 ${authMode==="register"?"tab-active":""}`}
                  onClick={() => setAuthMode("register")}>Register</button>
              </div>

              {authMode === "login" ? (
                <>
                  <h2 className="card-title text-2xl justify-center">Welcome Back</h2>
                  <label className="input input-bordered flex items-center gap-2">
                    <Mail className="size-4 opacity-70" />
                    <input type="email" className="grow" placeholder="Email"
                      value={email} onChange={e => setEmail(e.target.value)}
                      onKeyDown={e => e.key==="Enter" && handleLogin()} />
                  </label>
                  <label className="input input-bordered flex items-center gap-2">
                    <Lock className="size-4 opacity-70" />
                    <input type="password" className="grow" placeholder="Password"
                      value={password} onChange={e => setPassword(e.target.value)}
                      onKeyDown={e => e.key==="Enter" && handleLogin()} />
                  </label>
                  <div className="card-actions mt-2">
                    <button className="btn btn-primary w-full" onClick={handleLogin}>Login</button>
                  </div>
                  <p className="text-center text-xs text-base-content/40">
                    Admin: admin@admin.com / admin1234
                  </p>
                </>
              ) : (
                <>
                  <h2 className="card-title text-2xl justify-center">Create Account</h2>
                  <label className="input input-bordered flex items-center gap-2">
                    <UserIcon className="size-4 opacity-70" />
                    <input type="text" className="grow" placeholder="Username"
                      value={username} onChange={e => setUsername(e.target.value)}
                      onKeyDown={e => e.key==="Enter" && handleRegister()} />
                  </label>
                  <label className="input input-bordered flex items-center gap-2">
                    <Mail className="size-4 opacity-70" />
                    <input type="email" className="grow" placeholder="Email"
                      value={email} onChange={e => setEmail(e.target.value)}
                      onKeyDown={e => e.key==="Enter" && handleRegister()} />
                  </label>
                  <label className="input input-bordered flex items-center gap-2">
                    <Lock className="size-4 opacity-70" />
                    <input type="password" className="grow" placeholder="Password (min 8 chars)"
                      value={password} onChange={e => setPassword(e.target.value)}
                      onKeyDown={e => e.key==="Enter" && handleRegister()} />
                  </label>
                  <div className="card-actions mt-2">
                    <button className="btn btn-primary w-full" onClick={handleRegister}>
                      Create Account
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

        </div>
      </>
    );
  }

  // ── MAIN APP ─────────────────────────────────────────────
  return (
    <>
      <Toaster position="top-right" />
      <MainLayout
        response={response}       setResponse={setResponse}
        citations={citations}     setCitations={setCitations}
        latency={latency}         setLatency={setLatency}
        responseMethod={responseMethod}
        evalMetrics={evalMetrics}
        setEvalMetrics={setEvalMetrics}
        onSend={handleSend}       onLogout={handleLogout}
        loading={loading}         currentUser={currentUser}
        sessionId={sessionId}     setSessionId={setSessionId}
        method={method}           setMethod={setMethod}
        menuOpen={menuOpen}       setMenuOpen={setMenuOpen}
        showAdmin={showAdmin}     setShowAdmin={setShowAdmin}
        theme={theme}             onThemeToggle={toggleTheme}
        isAdmin={currentUser.role === "admin"}
      />
    </>
  );
}
