import { defineConfig } from 'vite';

// Em dev, a API do papel vem do servidor Python (python main.py --cad
// ou python -m draw.cad_server).
export default defineConfig({
  base: './',
  server: {
    proxy: { '/api': 'http://127.0.0.1:8770' },
  },
});
