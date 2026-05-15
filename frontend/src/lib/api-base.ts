export async function getApiBaseUrl(): Promise<string> {
  return import.meta.env.VITE_KEIBA_API_BASE_URL ?? 'http://127.0.0.1:8765';
}
