import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
var __dirname = fileURLToPath(new URL(".", import.meta.url));
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            three: path.resolve(__dirname, "node_modules/three/build/three.module.js"),
        },
    },
    optimizeDeps: {
        include: ["three"],
    },
    server: {
        allowedHosts: [".ngrok-free.app"],
        proxy: {
            "/api": "http://localhost:8000",
            "/live_detections": "http://localhost:8000",
            "/stream_agents": "http://localhost:8000",
            "/sessions": "http://localhost:8000",
            "/graph": "http://localhost:8000",
            "/health": "http://localhost:8000",
            "/getImage": "http://localhost:8000",
            "/exports": "http://localhost:8000",
        },
    },
});
