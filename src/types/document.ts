export interface DocumentRecord {
  document_id:  string;
  filename:     string;
  file_type:    string;
  status:       'processing' | 'completed' | 'failed';
  upload_date:  string;
}
