import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Non-Tauri environment (jsdom default — no __TAURI_INTERNALS__)
// ---------------------------------------------------------------------------
describe('getApiBaseUrl — non-Tauri environment', () => {
  beforeEach(() => {
    vi.resetModules();
    // Ensure __TAURI_INTERNALS__ is absent.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (window as any).__TAURI_INTERNALS__;
  });

  it('returns VITE_KEIBA_API_BASE_URL when set', async () => {
    vi.stubEnv('VITE_KEIBA_API_BASE_URL', 'http://127.0.0.1:9999');
    const { getApiBaseUrl } = await import('../lib/tauri');
    expect(await getApiBaseUrl()).toBe('http://127.0.0.1:9999');
    vi.unstubAllEnvs();
  });

  it('falls back to http://127.0.0.1:8765 when env var is not set', async () => {
    // Do NOT stub VITE_KEIBA_API_BASE_URL — let it be undefined.
    const { getApiBaseUrl } = await import('../lib/tauri');
    expect(await getApiBaseUrl()).toBe('http://127.0.0.1:8765');
  });
});

// ---------------------------------------------------------------------------
// Tauri environment (__TAURI_INTERNALS__ present)
// ---------------------------------------------------------------------------
describe('getApiBaseUrl — Tauri environment', () => {
  beforeEach(() => {
    vi.resetModules();
    // Simulate the Tauri WebView by injecting __TAURI_INTERNALS__.
    Object.defineProperty(window, '__TAURI_INTERNALS__', {
      value: {},
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (window as any).__TAURI_INTERNALS__;
    vi.resetModules();
  });

  it('calls invoke("get_api_port") and builds URL from the returned port', async () => {
    const mockInvoke = vi.fn().mockResolvedValue(12345);

    // Register the mock BEFORE importing tauri.ts so the dynamic import
    // inside getApiBaseUrl() picks it up.
    vi.doMock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));

    const { getApiBaseUrl } = await import('../lib/tauri');
    const url = await getApiBaseUrl();

    expect(mockInvoke).toHaveBeenCalledWith('get_api_port');
    expect(url).toBe('http://127.0.0.1:12345');
  });
});
