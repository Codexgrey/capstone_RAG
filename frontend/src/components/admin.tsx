import React, { useState, useEffect } from "react";
import { deleteDocument, listDocuments } from '../services/documentService';
import { UserPlus, Trash2, FileText, RefreshCw, Shield, User as UserIcon } from "lucide-react";
import toast from "react-hot-toast";

type User  = { username: string; password: string; role: "admin" | "user"; };
type Props = { onLogout: () => void; };

export default function Admin({ onLogout }: Props) {
  const [users,       setUsers]       = useState<User[]>([]);
  const [username,    setUsername]    = useState("");
  const [password,    setPassword]    = useState("");
  const [role,        setRole]        = useState<"admin"|"user">("user");
  const [documents,   setDocuments]   = useState<any[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("users");
    if (stored) setUsers(JSON.parse(stored));
    fetchDocuments();
  }, []);

  /* ── Documents ── */
  const fetchDocuments = async () => {
    setDocsLoading(true);
    try {
      const data = await listDocuments();
      setDocuments(data.documents || []);
    } catch {
      toast.error("Could not load documents");
    } finally {
      setDocsLoading(false);
    }
  };

  const handleDeleteDocument = async (documentId: string, filename: string) => {
    if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;
    try {
      await deleteDocument(documentId);
      toast.success(`${filename} deleted`);
      fetchDocuments();
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Delete failed");
    }
  };

  /* ── Users ── */
  const addUser = () => {
    if (!username || !password) { toast.error("Username and password required"); return; }
    const updated = [...users, { username, password, role }];
    localStorage.setItem("users", JSON.stringify(updated));
    setUsers(updated);
    setUsername(""); setPassword("");
    toast.success(`User ${username} added`);
  };

  const deleteUser = (index: number) => {
    const updated = users.filter((_, i) => i !== index);
    localStorage.setItem("users", JSON.stringify(updated));
    setUsers(updated);
    toast.success("User deleted");
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

      {/* ── Add User ── */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body gap-4">
          <h3 className="card-title flex items-center gap-2">
            <UserPlus className="size-5 text-primary" />
            Add User
          </h3>
          <input className="input input-bordered w-full" placeholder="Username"
            value={username} onChange={e => setUsername(e.target.value)} />
          <input className="input input-bordered w-full" placeholder="Password" type="password"
            value={password} onChange={e => setPassword(e.target.value)} />
          <select className="select select-bordered w-full" value={role}
            onChange={e => setRole(e.target.value as "admin"|"user")}>
            <option value="user">User</option>
            <option value="admin">Admin</option>
          </select>
          <div className="card-actions">
            <button className="btn btn-primary w-full" onClick={addUser}>Add User</button>
          </div>
        </div>
      </div>

      {/* ── User List ── */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200">
        <div className="card-body">
          <h3 className="card-title flex items-center gap-2">
            <Shield className="size-5 text-primary" />
            Users
          </h3>
          {users.length === 0 ? (
            <p className="text-base-content/40 text-sm italic">No users found</p>
          ) : (
            <ul className="flex flex-col gap-2 mt-2">
              {users.map((u, i) => (
                <li key={i} className="flex justify-between items-center bg-base-200 rounded-lg px-4 py-2">
                  <span className="flex items-center gap-2 text-base-content">
                    <UserIcon className="size-4 text-base-content/40" />
                    {u.username}
                    <span className={`badge badge-sm ${u.role==="admin" ? "badge-primary" : "badge-outline"}`}>
                      {u.role}
                    </span>
                  </span>
                  <button className="btn btn-error btn-outline btn-xs gap-1"
                    onClick={() => deleteUser(i)}>
                    <Trash2 className="size-3" /> Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* ── Documents List — full width ── */}
      <div className="card bg-base-100 border-t-4 border-[#00FF9D] hover:shadow-lg transition-all duration-200 md:col-span-2">
        <div className="card-body">
          <div className="flex items-center justify-between">
            <h3 className="card-title flex items-center gap-2">
              <FileText className="size-5 text-primary" />
              Uploaded Documents
            </h3>
            <button className="btn btn-ghost btn-sm gap-1" onClick={fetchDocuments}>
              <RefreshCw className="size-4" />
              Refresh
            </button>
          </div>

          {docsLoading ? (
            <div className="flex justify-center py-6">
              <span className="loading loading-dots loading-md text-primary" />
            </div>
          ) : documents.length === 0 ? (
            <p className="text-base-content/40 text-sm italic mt-2">
              No documents uploaded yet
            </p>
          ) : (
            <div className="overflow-x-auto mt-2">
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Uploaded</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc) => (
                    <tr key={doc.document_id} className="hover">
                      <td className="font-medium text-base-content max-w-xs truncate">
                        {doc.filename}
                      </td>
                      <td>
                        <span className="badge badge-ghost badge-sm uppercase">
                          {doc.file_type || "—"}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-sm ${
                          doc.status === "completed" ? "badge-success" :
                          doc.status === "failed"    ? "badge-error"   :
                          "badge-warning"
                        }`}>
                          {doc.status}
                        </span>
                      </td>
                      <td className="text-base-content/50 text-xs">
                        {new Date(doc.upload_date).toLocaleDateString()}
                      </td>
                      <td>
                        <button
                          className="btn btn-error btn-outline btn-xs gap-1"
                          onClick={() => handleDeleteDocument(doc.document_id, doc.filename)}
                        >
                          <Trash2 className="size-3" />
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}