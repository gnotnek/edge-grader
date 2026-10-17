import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { apiPlugin } from './server/api.ts'

export default defineConfig({
  plugins: [react(), tailwindcss(), apiPlugin(path.resolve(import.meta.dirname, '..'))],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'src'),
      '@server': path.resolve(import.meta.dirname, 'server'),
    },
  },
  // Local only: never reachable from another machine.
  server: { host: '127.0.0.1', port: 5173 },
})
