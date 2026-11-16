// After `vite build`: copy the WebAssembly runtime next to the app and write `dist/artifact.html`,
// the same page without <html>/<head>/<body> tags, for publishing as a claude.ai artifact.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

// base64 text copies of the binary models for hosts that only serve standard web types
import { existsSync } from "node:fs";
for (const f of ["features.onnx", "knn.onnx", "crack_seg.onnx", "tile_seg.onnx", "fabric_seg.onnx", "tile_memory.bin", "leather_memory.bin", "carpet_memory.bin", "wood_memory.bin"])
  if (existsSync(`dist/models/${f}`)) writeFileSync(`dist/models/${f}.b64.txt`, readFileSync(`dist/models/${f}`).toString("base64"));

const ORT = "node_modules/onnxruntime-web/dist/";
mkdirSync("dist/ort", { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) copyFileSync(ORT + f, "dist/ort/" + f);
const css = readFileSync("dist/assets/index.css", "utf8");
writeFileSync("dist/artifact.html", `<title>ForgeSight Inspector</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="fs-model-encoding" content="base64">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>${css}</style>
<div id="app"></div>
<script type="module" src="assets/app.js"></script>
`);
console.log("dist ready: index.html (any static host) + artifact.html (claude.ai)");
