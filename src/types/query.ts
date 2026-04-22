export interface Citation {
  chunk_id:    string;
  source_name: string;
  page:        number | null;
  section:     string | null;
}

export interface QueryResponse {
  answer:           string;
  citations:        Citation[];
  retrieval_method: string;
  latency_ms:       number;
  session_id:       string;
}
