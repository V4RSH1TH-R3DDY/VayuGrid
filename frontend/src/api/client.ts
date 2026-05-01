const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type LoginResponse = {
  access_token: string;
  token_type: string;
  role: string;
  node_id?: number | null;
};

export const authStore = {
  getToken: () => localStorage.getItem("vg_token"),
  getRole: () => localStorage.getItem("vg_role"),
  getNodeId: () => localStorage.getItem("vg_node_id"),
  setAuth: (token: string, role: string, nodeId?: number | null) => {
    localStorage.setItem("vg_token", token);
    localStorage.setItem("vg_role", role);
    if (nodeId !== undefined && nodeId !== null) {
      localStorage.setItem("vg_node_id", String(nodeId));
    }
  },
  clear: () => {
    localStorage.removeItem("vg_token");
    localStorage.removeItem("vg_role");
    localStorage.removeItem("vg_node_id");
  },
};

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers || {});
  const token = authStore.getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export async function login(username: string, password: string): Promise<LoginResponse> {
  return apiFetch<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}
