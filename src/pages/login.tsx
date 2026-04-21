import React, { useState } from "react";

type User = {
  username: string;
  password: string;
  role: "admin" | "user";
};

type Props = {
  onLogin: (user: User) => void;
};

export default function Login({ onLogin }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = () => {
    const users: User[] = JSON.parse(localStorage.getItem("users") || "[]");

    const user = users.find(
      (u) => u.username === username && u.password === password
    );

    if (!user) {
      alert("Invalid credentials");
      return;
    }

    localStorage.setItem("currentUser", JSON.stringify(user));
    onLogin(user);
  };

  return (
    <div className="centered">
      <div className="login-card">
        <h2>Login</h2>

        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button onClick={handleLogin}>Login</button>
      </div>
    </div>
  );
}