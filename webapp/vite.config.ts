import { defineConfig } from "vite";

// Relative paths everywhere so the build works from any folder or host (claude.ai, GitHub Pages, Netlify, a USB stick).
export default defineConfig({
  base: "./",
  build: {
    target: "es2020",
    assetsInlineLimit: 0,
    rollupOptions: { output: { entryFileNames: "assets/app.js", chunkFileNames: "assets/[name].js", assetFileNames: "assets/[name][extname]" } },
  },
});
