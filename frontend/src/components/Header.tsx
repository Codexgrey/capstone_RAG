import React from "react";
import { LogOut, Menu, MessageSquare, Settings, Sun, Moon } from "lucide-react";

type HeaderProps = {
  setMenuOpen:    (open: boolean) => void;
  showMenu?:      boolean;
  username?:      string;
  onLogout?:      () => void;
  onAdminToggle?: () => void;
  showAdmin?:     boolean;
  theme:          "forest" | "raglight";
  onThemeToggle:  () => void;
};

export default function Header({
  setMenuOpen, showMenu = true, username, onLogout,
  onAdminToggle, showAdmin, theme, onThemeToggle,
}: HeaderProps) {
  const isDark = theme === "forest";

  return (
    <header className="bg-base-300 border-b border-base-content/10 sticky top-0 z-10">
      <div className="mx-auto max-w-7xl px-4 py-3">
        <div className="flex items-center justify-between">

          {/* LEFT */}
          <div className="flex items-center gap-3">
            {showMenu && (
              <button className="btn btn-ghost btn-circle btn-sm"
                onClick={() => setMenuOpen(true)} title="Open controls">
                <Menu className="size-5" />
              </button>
            )}
            <div className="flex items-center gap-2">
              <MessageSquare className="size-6 text-primary" />
              <h1 className="text-2xl font-bold text-primary font-mono tracking-tight">
                RAG System
              </h1>
            </div>
          </div>

          {/* RIGHT */}
          <div className="flex items-center gap-2">

            {/* Admin toggle */}
            {onAdminToggle && (
              <button className="btn btn-ghost btn-sm gap-2" onClick={onAdminToggle}>
                <Settings className="size-4" />
                <span className="hidden sm:block">
                  {showAdmin ? "Back to App" : "Admin Panel"}
                </span>
              </button>
            )}

            {/* Light / Dark toggle */}
            <button
              className="btn btn-ghost btn-circle btn-sm"
              onClick={onThemeToggle}
              title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDark
                ? <Sun  className="size-4 text-yellow-400" />
                : <Moon className="size-4 text-primary" />
              }
            </button>

            {/* Username */}
            {username && (
              <span className="text-base-content/60 text-sm hidden sm:block">{username}</span>
            )}

            {/* Logout */}
            {onLogout && (
              <button className="btn btn-error btn-outline btn-sm gap-2" onClick={onLogout}>
                <LogOut className="size-4" />
                <span className="hidden sm:block">Log Out</span>
              </button>
            )}
          </div>

        </div>
      </div>
    </header>
  );
}