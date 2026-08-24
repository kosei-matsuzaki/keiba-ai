import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
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
    // 既定の 5 秒だと、全ファイルを並列実行したときに jsdom の環境構築が
    // 重なって取りこぼす (単体実行では通るのに全体実行だけ落ちる)。
    testTimeout: 20_000,
    hookTimeout: 20_000,
  },
});
