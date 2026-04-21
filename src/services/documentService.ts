import apiClient from './apiClient';

export const uploadDocument = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await apiClient.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const listDocuments = async () => {
  const res = await apiClient.get('/documents');
  return res.data;
};

export const getDocumentStatus = async (documentId: string) => {
  const res = await apiClient.get(`/document/${documentId}/status`);
  return res.data;
};
