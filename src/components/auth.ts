export type User = {
  username: string;
  password: string;
  role: "admin" | "user";
};

export const getUsers = (): User[] => {
  const users = localStorage.getItem("users");
  return users ? JSON.parse(users) : [];
};

export const saveUsers = (users: User[]) => {
  localStorage.setItem("users", JSON.stringify(users));
};

export const initUsers = () => {
  if (!localStorage.getItem("users")) {
    saveUsers([
      { username: "admin", password: "admin", role: "admin" },
      { username: "user", password: "1234", role: "user" }
    ]);
  }
};

export const loginUser = (username: string, password: string) => {
  const users = getUsers();
  return users.find(u => u.username === username && u.password === password);
};

export const setCurrentUser = (user: User) => {
  localStorage.setItem("currentUser", JSON.stringify(user));
};

export const getCurrentUser = (): User | null => {
  const user = localStorage.getItem("currentUser");
  return user ? JSON.parse(user) : null;
};

export const logout = () => {
  localStorage.removeItem("currentUser");
};