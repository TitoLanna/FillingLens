import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // In dev, /v1/* goes to the API gateway (backend/gateway, port 8080), never straight to the
    // backend, so rate limits and validation apply in development too. One origin: no CORS needed.
    proxy: { "/v1": { target: "http://localhost:8080", changeOrigin: true } },
  },
});
