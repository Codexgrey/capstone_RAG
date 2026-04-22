import React from "react";

type HeaderProps = {
  setMenuOpen: (open: boolean) => void;
  showMenu?: boolean;
};

export default function Header({
  setMenuOpen,
  showMenu = true
}: HeaderProps) {
  return (
    <header className="header">
      
      {/* SHOW ONLY IF TRUE */}
      {showMenu && (
        <button
          className="menu-btn"
          onClick={() => setMenuOpen(true)}
        >
          ☰
        </button>
      )}

      <div className="title-container">
        <img
          src="/logo.jpg"
          alt="Logo"
          className="logo"
        />

        <h1>RAG System</h1>
      </div>
    </header>
  );
}