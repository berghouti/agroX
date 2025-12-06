import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  build: {
    target: 'esnext'
  },
  css: {
    modules: false,
    preprocessorOptions: {
      css: {
        additionalData: `@import "./src/styles/theme.css"; @import "./src/styles/utilities.css";`
      }
    }
  }
});