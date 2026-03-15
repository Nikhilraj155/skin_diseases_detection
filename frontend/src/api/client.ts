const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Request failed.");
  }
  return response.json() as Promise<T>;
}

export { API_BASE_URL };
