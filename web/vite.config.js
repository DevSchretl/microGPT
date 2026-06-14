import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the FastAPI backend so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
