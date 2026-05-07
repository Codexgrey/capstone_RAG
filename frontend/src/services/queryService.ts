import apiClient from './apiClient';

export const sendQuery = async (
  question: string,
  sessionId?: string,
  retrievalMethod: string = 'vector'
) => {
  const res = await apiClient.post('/api/query', {
    question,
    session_id:       sessionId || null,
    retrieval_method: retrievalMethod,
    top_k:            5
  });
  return res.data;
};

export const getSessions = async () => {
  const res = await apiClient.get('/api/chat/sessions');
  return res.data;
};

export const getSessionMessages = async (sessionId: string) => {
  const res = await apiClient.get(`/api/chat/sessions/${sessionId}`);
  return res.data;
};

export const renameSession = async (id: string, title: string) => {
  const res = await apiClient.patch(`/api/chat/sessions/${id}`, { title });
  return res.data;
};

export const deleteSession = async (id: string) => {
  const res = await apiClient.delete(`/api/chat/sessions/${id}`);
  return res.data;
};