/**
 * Tauri environment detection and API base URL resolution.
 *
 * In a Tauri WebView `window.__TAURI_INTERNALS__` is injected by the runtime.
 * In a plain browser (dev server, test runner) it is absent.
 */

const isTauri =
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

/**
 * Return the base URL for the FastAPI backend.
 *
 * - Tauri production: invoke 'get_api_port' to get the dynamically assigned
 *   port chosen at startup, then construct http://127.0.0.1:<port>.
 * - Browser / dev server: fall back to VITE_KEIBA_API_BASE_URL env var or
 *   the hardcoded development default.
 */
export async function getApiBaseUrl(): Promise<string> {
  if (isTauri) {
    const { invoke } = await import('@tauri-apps/api/core');
    const port = await invoke<number>('get_api_port');
    return `http://127.0.0.1:${port}`;
  }
  return import.meta.env.VITE_KEIBA_API_BASE_URL ?? 'http://127.0.0.1:8765';
}

export { isTauri };
