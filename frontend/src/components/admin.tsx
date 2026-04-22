import React, { useState, useEffect } from "react";

type User = {
  username: string;
  password: string;
  role: "admin" | "user";
};

type Props = {
  onLogout: () => void;
};

export default function Admin({ onLogout }: Props) {
  const [users, setUsers] = useState<User[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"admin" | "user">("user");

  /* LOAD USERS */
  useEffect(() => {
    const stored = localStorage.getItem("users");
    if (stored) setUsers(JSON.parse(stored));
  }, []);

  /* ADD USER */
  const addUser = () => {
    if (!username || !password) return;

    const updated = [...users, { username, password, role }];
    localStorage.setItem("users", JSON.stringify(updated));
    setUsers(updated);

    setUsername("");
    setPassword("");
  };

  /* DELETE USER */
  const deleteUser = (index: number) => {
    const updated = users.filter((_, i) => i !== index);
    localStorage.setItem("users", JSON.stringify(updated));
    setUsers(updated);
  };

  return (
    <div className="admin-page">
      <div className="admin-header">
        <h2>Admin Panel</h2>
        <button onClick={onLogout}>Logout</button>
      </div>

      <div className="admin-container">

        {/* ADD USER */}
        <div className="admin-card">
          <h3>Add User</h3>

          <input
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />

          <input
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <select
            value={role}
            onChange={(e) => setRole(e.target.value as "admin" | "user")}
          >
            <option value="user">User</option>
            <option value="admin">Admin</option>
          </select>

          <button onClick={addUser}>Add User</button>
        </div>

        {/* USER LIST */}
        <div className="admin-card">
          <h3>Users</h3>

          {users.length === 0 ? (
            <p>No users found</p>
          ) : (
            <ul className="user-list">
              {users.map((u, i) => (
                <li key={i}>
                  <span>
                    {u.username} ({u.role})
                  </span>
                  <button onClick={() => deleteUser(i)}>Delete</button>
                </li>
              ))}
            </ul>
          )}
        </div>

      </div>
    </div>
  );
}