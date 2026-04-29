/**
 * Stub for @tauri-apps/api/core used in the Vitest environment.
 * The real package is only available at runtime inside a Tauri WebView.
 * Tests that exercise the Tauri code path use vi.doMock to override this stub.
 */
export async function invoke<T = unknown>(_cmd: string, ..._args: unknown[]): Promise<T> {
  throw new Error(
    '@tauri-apps/api/core is not available in this environment. ' +
      'Use vi.doMock to override invoke in tests that exercise the Tauri path.'
  );
}
