import React from "react";

interface ResponseCardProps {
  response: string;
}

const ResponseCard: React.FC<ResponseCardProps> = ({ response }) => {
  return (
    <div className="response-card">
      <h3>Response</h3>
      <p>{response || "Ask a question to see the RAG system response."}</p>
    </div>
  );
};

export default ResponseCard;