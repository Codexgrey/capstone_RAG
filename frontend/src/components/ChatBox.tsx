import React, { useState } from "react";

interface ChatBoxProps {
  onSend: (query: string) => void;
}

const ChatBox: React.FC<ChatBoxProps> = ({ onSend }) => {
  const [query, setQuery] = useState("");

  const handleSend = () => {
    if (!query.trim()) return;
    onSend(query);
    setQuery("");
  };

  return (
    <div className="chat-box">
      <input
        className="chat-input"
        placeholder="Ask a question..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <button className="send-btn" onClick={handleSend}>
        ➤
      </button>
    </div>
  );
};

export default ChatBox;