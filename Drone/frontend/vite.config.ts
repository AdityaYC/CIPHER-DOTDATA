import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [".ngrok-free.app"],
    proxy: {
      "/api": "http://localhost:8000",
      "/stream_agents": "http://localhost:8000",
      "/sessions": "http://localhost:8000",
      "/graph": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/getImage": "http://localhost:8000",
    },
  },
});

