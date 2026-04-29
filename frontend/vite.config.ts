import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      // In the Vitest / browser-dev environment the Tauri runtime package is
      // not installed. Map it to a local stub so imports resolve without error.
      // Tests that exercise the Tauri invoke path use vi.doMock to override it.
      '@tauri-apps/api/core': path.resolve(__dirname, './src/__mocks__/tauri-api-core.ts'),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
  // @ts-expect-error vitest injects test config via its own type augmentation
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
    globals: true,
  },
});
