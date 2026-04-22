import React, { useState, useEffect } from "react";
import "./App.css";

import Sidebar from "../components/SideBar";
import ChatBox from "../components/ChatBox";
import ResponseCard from "../components/ResponseCard";
import SourcesPanel from "../components/SourcesPanel";
import Header from "../components/Header";

import { sendQuery } from "../services/queryService";

/* ================= TYPES ================= */
type User = {
  username: string;
  password: string;
  role: "admin" | "user";
};

type Citation = {
  chunk_id: string;
  source_name: string;
  page: number;
  section: string | null;
};

type Latency = {
  retrieval: number;
  llm: number;
};

/* =====================================================
   USER MAIN PAGE
===================================================== */
function UserMainPage({
  response,
  citations,
  latency,
  onSend,
  onLogout
}: {
  response: string;
  citations: Citation[];
  latency: Latency;
  onSend: (query: string) => void;
  onLogout: () => void;
}) {
  return (
    <div className="app">
      <div className="layout">
        <main className="main">
          <Header setMenuOpen={() => {}} showMenu={false} />

          <div
            style={{
              position: "absolute",
              right: 20,
              top: 20
            }}
          >
            <button onClick={onLogout}>Logout</button>
          </div>

          <ResponseCard response={response} />
          <ChatBox onSend={onSend} />
        </main>

        <SourcesPanel
          citations={citations}
          latency_ms={latency.retrieval}
        />
      </div>
    </div>
  );
}

/* =====================================================
   ADMIN MAIN PAGE
===================================================== */
function AdminMainPage({
  menuOpen,
  setMenuOpen,
  response,
  citations,
  latency,
  onSend,
  onLogout,
  showAdmin,
  setShowAdmin,
  users,
  newUser,
  setNewUser,
  newPass,
  setNewPass,
  role,
  setRole,
  addUser,
  deleteUser,
  setMethod
}: any) {
  return (
    <div className="app">
      <Sidebar
        open={menuOpen}
        setOpen={setMenuOpen}
        setMethod={setMethod}
      />

      <div className="layout">
        <main className="main">
          <Header setMenuOpen={setMenuOpen} showMenu={true} />

          <div
            style={{
              position: "absolute",
              right: 20,
              top: 20,
              display: "flex",
              gap: 10
            }}
          >
            <button onClick={() => setShowAdmin(!showAdmin)}>
              {showAdmin ? "Back to App" : "Admin Panel"}
            </button>

            <button onClick={onLogout}>Logout</button>
          </div>

          {!showAdmin ? (
            <>
              <ResponseCard response={response} />
              <ChatBox onSend={onSend} />
            </>
          ) : (
            <div className="admin-container">
              <div className="admin-card">
                <h3>Add User</h3>

                <input
                  placeholder="Username"
                  value={newUser}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setNewUser(e.target.value)
                  }
                />

                <input
                  placeholder="Password"
                  value={newPass}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setNewPass(e.target.value)
                  }
                />

                <select
                  value={role}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                    setRole(e.target.value)
                  }
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                </select>

                <button onClick={addUser}>Add User</button>
              </div>

              <div className="admin-card">
                <h3>Users</h3>

                <ul className="user-list">
                  {users.map((u: User, i: number) => (
                    <li key={i}>
                      <span>
                        {u.username} ({u.role})
                      </span>

                      <button onClick={() => deleteUser(i)}>
                        Delete
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </main>

        <SourcesPanel
          citations={citations}
          latency_ms={latency.retrieval}
        />
      </div>
    </div>
  );
}

/* =====================================================
   APP
===================================================== */
export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  const [userResponse, setUserResponse] = useState("");
  const [adminResponse, setAdminResponse] = useState("");

  const [citations, setCitations] = useState<Citation[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [latency, setLatency] = useState<Latency>({
    retrieval: 0,
    llm: 0
  });

  const [method, setMethod] = useState("keyword");

  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [users, setUsers] = useState<User[]>([]);

  const [showAdmin, setShowAdmin] = useState(false);

  /* LOGIN */
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  /* ADMIN */
  const [newUser, setNewUser] = useState("");
  const [newPass, setNewPass] = useState("");
  const [role, setRole] = useState<"admin" | "user">("user");

  /* INIT USERS */
  useEffect(() => {
    const stored = localStorage.getItem("users");

    if (!stored) {
      const defaults: User[] = [
        {
          username: "admin",
          password: "admin",
          role: "admin"
        },
        {
          username: "user",
          password: "1234",
          role: "user"
        }
      ];

      localStorage.setItem("users", JSON.stringify(defaults));
      setUsers(defaults);
    } else {
      setUsers(JSON.parse(stored));
    }

    const saved = localStorage.getItem("currentUser");
    if (saved) setCurrentUser(JSON.parse(saved));
  }, []);

  /* LOGIN */
  const handleLogin = () => {
    const user = users.find(
      (u) =>
        u.username === username &&
        u.password === password
    );

    if (!user) {
      alert("Invalid credentials");
      return;
    }

    localStorage.setItem("currentUser", JSON.stringify(user));
    setCurrentUser(user);
  };

  /* LOGOUT */
  const handleLogout = () => {
    localStorage.removeItem("currentUser");
    setCurrentUser(null);
    setShowAdmin(false);
    setSessionId(null);
  };

  /* ================= CHAT ================= */

  const handleUserSend = async (query: string) => {
    try {
      const result = await sendQuery(
        query,
        sessionId || undefined,
        method
      );

      setUserResponse(result.answer);
      setSessionId(result.session_id);
      setCitations(result.citations);

      setLatency({
        retrieval: result.latency_ms,
        llm: 0
      });
    } catch {
      setUserResponse("Error contacting backend.");
    }
  };

  const handleAdminSend = async (query: string) => {
    try {
      const result = await sendQuery(
        query,
        sessionId || undefined,
        method
      );

      setAdminResponse(result.answer);
      setSessionId(result.session_id);
      setCitations(result.citations);

      setLatency({
        retrieval: result.latency_ms,
        llm: 0
      });
    } catch {
      setAdminResponse("Error contacting backend.");
    }
  };

  /* ================= ADMIN USERS ================= */

  const addUser = () => {
    if (!newUser || !newPass) return;

    const updated = [
      ...users,
      {
        username: newUser,
        password: newPass,
        role
      }
    ];

    setUsers(updated);
    localStorage.setItem("users", JSON.stringify(updated));

    setNewUser("");
    setNewPass("");
  };

  const deleteUser = (index: number) => {
    const updated = users.filter((_, i) => i !== index);

    setUsers(updated);
    localStorage.setItem("users", JSON.stringify(updated));
  };

  /* ================= LOGIN PAGE ================= */

if (!currentUser) {
  return (
    <div className="centered">

      {/* PAGE HEADER */}
      <div className="login-page-header">
        <img
          src="/logo.jpg"
          alt="Logo"
          className="login-logo"
        />

        <h1 className="login-title">
          RAG SYSTEM
        </h1>
      </div>

      {/* LOGIN CARD */}
      <div className="login-card">
        <h2>Login</h2>

        <input
          placeholder="Username"
          value={username}
          onChange={(e) =>
            setUsername(e.target.value)
          }
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) =>
            setPassword(e.target.value)
          }
        />

        <button onClick={handleLogin}>
          Login
        </button>
      </div>

    </div>
  );
}

  /* ================= USER PAGE ================= */

  if (currentUser.role === "user") {
    return (
      <UserMainPage
        response={userResponse}
        citations={citations}
        latency={latency}
        onSend={handleUserSend}
        onLogout={handleLogout}
      />
    );
  }

  /* ================= ADMIN PAGE ================= */

  return (
    <AdminMainPage
      menuOpen={menuOpen}
      setMenuOpen={setMenuOpen}
      response={adminResponse}
      citations={citations}
      latency={latency}
      onSend={handleAdminSend}
      onLogout={handleLogout}
      showAdmin={showAdmin}
      setShowAdmin={setShowAdmin}
      users={users}
      newUser={newUser}
      setNewUser={setNewUser}
      newPass={newPass}
      setNewPass={setNewPass}
      role={role}
      setRole={setRole}
      addUser={addUser}
      deleteUser={deleteUser}
      setMethod={setMethod}
    />
  );
}