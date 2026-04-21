import apiClient from './apiClient';

export const register = async (username: string, email: string, password: string) => {
  const res = await apiClient.post('/api/auth/register', {
    username, email, password
  });
  return res.data;
};

export const login = async (email: string, password: string) => {
  const res = await apiClient.post('/api/auth/login', {
    email, password
  });
  // Save token so apiClient interceptor picks it up
  localStorage.setItem('token', res.data.access_token);
  return res.data;
};

export const logout = () => {
  localStorage.removeItem('token');
};

export const isLoggedIn = () => {
  return !!localStorage.getItem('token');
};
