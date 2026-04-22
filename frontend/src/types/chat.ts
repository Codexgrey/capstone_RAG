export interface ChatMessage {
  id:               string;
  role:             'user' | 'assistant';
  content:          string;
  created_at:       string;
  retrieval_method: string | null;
}

export interface ChatSession {
  session_id:  string;
  title:       string;
  created_at:  string;
  updated_at:  string;
}
