import apiClient from './apiClient';

export const sendQuery = async (
  question: string,
  sessionId?: string,
  retrievalMethod: string = 'vector'
) => {
  const res = await apiClient.post('/query', {
    question,
    session_id:       sessionId || null,
    retrieval_method: retrievalMethod,
    top_k:            5
  });
  return res.data;
};

export const getSessions = async () => {
  const res = await apiClient.get('/chat/sessions');
  return res.data;
};
export const getSessionMessages = async (sessionId: string) => {
  const res = await apiClient.get(`/chat/sessions/${sessionId}`);
  return res.data;
};
