import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Tauri runtime package のスタブは Vitest 実行時のみ適用する。
// production / dev build では Tauri WebView が本物の @tauri-apps/api/core を
// 解決できないと invoke('get_api_port') が落ちて API 呼び出しが全滅するため。
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      ...(mode === 'test'
        ? {
            '@tauri-apps/api/core': path.resolve(
              __dirname,
              './src/__mocks__/tauri-api-core.ts'
            ),
          }
        : {}),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
    globals: true,
  },
}));
