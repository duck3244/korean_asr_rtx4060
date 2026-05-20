import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 개발 시 /api, /health 요청을 FastAPI 백엔드(:8000)로 프록시 → CORS 불필요
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
});
