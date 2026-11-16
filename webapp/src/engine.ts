// In-browser inspection engine. Everything here runs on the viewer's machine: no image ever leaves the browser.
// It mirrors backend/app/ml (PatchCore, segmentation, decision rules) using the ONNX exports in public/models.
import * as ort from "onnxruntime-web/wasm";

export type Outcome = "pass" | "clear" | "review" | "defective";
export type Mode = "learn" | "crack" | "tile";
export type Sensitivity = "low" | "normal" | "high";

export interface Finding {
  type: string;
  label: string;
  category: string;
  typeSource: "model" | "heuristic";
  typeConf: number | null;
  conf: number;
  basis: string;
  box: [number, number, number, number]; // x, y, w, h in model-input pixels
  severity: "minor" | "major" | "critical";
  location: string;
  causes: string[];
}

export interface Result {
  outcome: Outcome;
  pDefect: number | null;
  reasons: string[];
  findings: Finding[];
  mask: Uint8Array; // finding index + 1 per pixel of the model input (0 = none)
  maskW: number;
  maskH: number;
  ms: number;
  quality: "ok" | "warn" | "fail";
}

interface Config {
  tile: {
    segInput: [number, number]; pcInput: [number, number]; classes: string[]; segPixelThreshold: number; memoryRows: number;
    fusion: { coef: [number, number]; intercept: number; t_low: number; t_high: number };
    platt: Record<string, { a: number; b: number }>; disagreementDelta: number;
  };
  crack: { segInput: [number, number]; classes: string[]; segPixelThreshold: number; regionConfident: number };
  materials: { key: string; name: string; family: string; imaging: string }[];
  taxonomy: Record<string, { label: string; category: string; crit: number }>;
  causes: Record<string, string[]>;
}

export let config: Config;
let sFeat: ort.InferenceSession;
let sKnn: ort.InferenceSession;
const lazy: Record<string, Promise<ort.InferenceSession>> = {};
let tileMemory: ort.Tensor | null = null;

const url = (p: string) => new URL(p, document.baseURI).href;
// Some hosts (claude.ai artifacts) only serve standard web file types, so the page can ask for base64 text copies of the models.
const B64 = document.querySelector('meta[name="fs-model-encoding"]')?.getAttribute("content") === "base64";
async function fetchBuf(p: string): Promise<ArrayBuffer> {
  const r = await fetch(url(B64 ? `${p}.b64.txt` : p));
  if (!r.ok) throw new Error(`Could not load ${p} (HTTP ${r.status})`);
  if (!B64) return r.arrayBuffer();
  const bin = atob((await r.text()).trim());
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

export async function loadEngine(onStep: (msg: string) => void) {
  ort.env.wasm.wasmPaths = url("ort/");
  ort.env.wasm.numThreads = 1; // no cross-origin isolation in a plain static host or sandboxed frame
  onStep("Loading configuration");
  config = await (await fetch(url("models/config.json"))).json();
  onStep("Loading feature extractor");
  sFeat = await ort.InferenceSession.create(await fetchBuf("models/features.onnx"));
  sKnn = await ort.InferenceSession.create(await fetchBuf("models/knn.onnx"));
}

function session(name: "crack_seg" | "tile_seg"): Promise<ort.InferenceSession> {
  lazy[name] ??= fetchBuf(`models/${name}.onnx`).then((b) => ort.InferenceSession.create(b));
  return lazy[name];
}

async function tileMem(): Promise<ort.Tensor> {
  if (!tileMemory) tileMemory = new ort.Tensor("float32", new Float32Array(await fetchBuf("models/tile_memory.bin")), [config.tile.memoryRows, 128]);
  return tileMemory;
}

// ------------------------------------------------------------------ image helpers
const scratch = document.createElement("canvas");
const sctx = scratch.getContext("2d", { willReadFrequently: true })!;

export interface Pixels { rgb: Float32Array; gray: Float32Array; w: number; h: number }

/** Axis weights of OpenCV's INTER_AREA resize (cv::resize), so the models see the same pixels as in training. */
type Taps = { i0: Int32Array; n: Uint8Array; idx: Int32Array; wt: Float32Array };
function areaTaps(src: number, dst: number, zoomPath: boolean): Taps {
  const scale = src / dst;
  const i0 = new Int32Array(dst), n = new Uint8Array(dst);
  const idx: number[] = [], wt: number[] = [];
  for (let d = 0; d < dst; d++) {
    i0[d] = idx.length;
    if (zoomPath) {
      // cv::resize INTER_AREA when either axis is enlarged: two taps with OpenCV's own coefficient formula
      const sx = Math.floor(d * scale);
      let f = d + 1 - (sx + 1) / scale;
      f = f <= 0 ? 0 : f - Math.floor(f);
      const a = Math.min(Math.max(sx, 0), src - 1), b = Math.min(Math.max(sx + 1, 0), src - 1);
      idx.push(a, b); wt.push(1 - f, f); n[d] = 2;
    } else {
      // true area averaging: overlap of [d*scale, (d+1)*scale) with each source pixel
      const f0 = d * scale, f1 = (d + 1) * scale;
      let c = 0;
      for (let s = Math.floor(f0); s < Math.ceil(f1 - 1e-9) && s < src; s++) {
        const w = (Math.min(f1, s + 1) - Math.max(f0, s)) / scale;
        if (w > 1e-9) { idx.push(s); wt.push(w); c++; }
      }
      n[d] = c;
    }
  }
  return { i0, n, idx: Int32Array.from(idx), wt: Float32Array.from(wt) };
}

/** Resize a frame to (w, h) like cv2.resize(INTER_AREA) on 8-bit pixels; planar RGB 0..1 plus grayscale 0..255. */
export function pixels(src: CanvasImageSource, w: number, h: number): Pixels {
  const sw = (src as HTMLVideoElement).videoWidth || (src as unknown as { width: number }).width;
  const sh = (src as HTMLVideoElement).videoHeight || (src as unknown as { height: number }).height;
  scratch.width = sw;
  scratch.height = sh;
  sctx.drawImage(src, 0, 0, sw, sh);
  const d = sctx.getImageData(0, 0, sw, sh).data;
  const zoom = !(sw / w >= 1 && sh / h >= 1);
  const tx = areaTaps(sw, w, zoom), ty = areaTaps(sh, h, zoom);
  const n = w * h;
  const rgb = new Float32Array(3 * n);
  const gray = new Float32Array(n);
  const row = new Float32Array(sw * 3);
  const tmp = new Float32Array(w * 3 * h);
  // horizontal pass per source row is wasteful for area downscale, so do vertical first into row buffers
  for (let y = 0; y < h; y++) {
    row.fill(0);
    for (let k = 0; k < ty.n[y]; k++) {
      const sy = ty.idx[ty.i0[y] + k], wy = ty.wt[ty.i0[y] + k];
      const o = sy * sw * 4;
      for (let x = 0; x < sw; x++) { row[3 * x] += wy * d[o + 4 * x]; row[3 * x + 1] += wy * d[o + 4 * x + 1]; row[3 * x + 2] += wy * d[o + 4 * x + 2]; }
    }
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < tx.n[x]; k++) {
        const sx = tx.idx[tx.i0[x] + k], wx = tx.wt[tx.i0[x] + k];
        r += wx * row[3 * sx]; g += wx * row[3 * sx + 1]; b += wx * row[3 * sx + 2];
      }
      tmp[3 * (y * w + x)] = r; tmp[3 * (y * w + x) + 1] = g; tmp[3 * (y * w + x) + 2] = b;
    }
  }
  for (let i = 0; i < n; i++) {
    const r = Math.min(255, Math.max(0, Math.round(tmp[3 * i]))), g = Math.min(255, Math.max(0, Math.round(tmp[3 * i + 1]))), b = Math.min(255, Math.max(0, Math.round(tmp[3 * i + 2])));
    rgb[i] = r / 255; rgb[n + i] = g / 255; rgb[2 * n + i] = b / 255;
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return { rgb, gray, w, h };
}

const imageTensor = (p: Pixels) => new ort.Tensor("float32", p.rgb, [1, 3, p.h, p.w]);

/** OpenCV-compatible separable Gaussian blur (float kernel size, BORDER_REFLECT_101). */
export function gaussian(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  let k = Math.round(sigma * 8 + 1) | 1;
  if (k < 3) k = 3;
  const r = k >> 1;
  const ker = new Float32Array(k);
  let s = 0;
  for (let i = 0; i < k; i++) { ker[i] = Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)); s += ker[i]; }
  for (let i = 0; i < k; i++) ker[i] /= s;
  const refl = (i: number, n: number) => {
    if (n === 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  };
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let j = 0; j < k; j++) a += ker[j] * src[y * w + refl(x + j - r, w)];
      tmp[y * w + x] = a;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let j = 0; j < k; j++) a += ker[j] * tmp[refl(y + j - r, h) * w + x];
      out[y * w + x] = a;
    }
  return out;
}

/** Bilinear resize, same sampling as cv2.INTER_LINEAR. */
export function resize(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  const out = new Float32Array(dw * dh);
  const fx = sw / dw, fy = sh / dh;
  for (let y = 0; y < dh; y++) {
    let sy = (y + 0.5) * fy - 0.5;
    sy = Math.min(Math.max(sy, 0), sh - 1);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, sh - 1), wy = sy - y0;
    for (let x = 0; x < dw; x++) {
      let sx = (x + 0.5) * fx - 0.5;
      sx = Math.min(Math.max(sx, 0), sw - 1);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, sw - 1), wx = sx - x0;
      out[y * dw + x] = (1 - wy) * ((1 - wx) * src[y0 * sw + x0] + wx * src[y0 * sw + x1]) + wy * ((1 - wx) * src[y1 * sw + x0] + wx * src[y1 * sw + x1]);
    }
  }
  return out;
}

interface Component { idx: Int32Array; x0: number; y0: number; x1: number; y1: number; peak: number }

/** 8-connected components of `score > thr`, largest first. */
function components(score: Float32Array, w: number, h: number, thr: number, minArea: number, maxN = 40): Component[] {
  const seen = new Uint8Array(w * h);
  const out: Component[] = [];
  const stack = new Int32Array(w * h);
  for (let s = 0; s < w * h; s++) {
    if (seen[s] || !(score[s] > thr)) continue;
    let sp = 0, n = 0;
    const pix: number[] = [];
    stack[sp++] = s; seen[s] = 1;
    let x0 = w, y0 = h, x1 = 0, y1 = 0, peak = 0;
    while (sp) {
      const p = stack[--sp];
      pix.push(p); n++;
      const px = p % w, py = (p / w) | 0;
      if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
      if (score[p] > peak) peak = score[p];
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (!seen[q] && score[q] > thr) { seen[q] = 1; stack[sp++] = q; }
        }
    }
    if (n >= minArea) out.push({ idx: Int32Array.from(pix), x0, y0, x1, y1, peak });
  }
  out.sort((a, b) => b.idx.length - a.idx.length);
  return out.slice(0, maxN);
}

// ------------------------------------------------------------------ shared post-processing
function location(cx: number, cy: number, w: number, h: number) {
  const col = ["left", "centre", "right"][Math.min(2, Math.floor((3 * cx) / w))];
  const row = ["top", "middle", "bottom"][Math.min(2, Math.floor((3 * cy) / h))];
  return row === "middle" && col === "centre" ? "centre" : `${row}-${col}`;
}

function severity(type: string, areaRatio: number, extent: number): Finding["severity"] {
  const crit = config.taxonomy[type]?.crit ?? 0.5;
  if (["open", "short", "missing_component", "break"].includes(type)) return "critical";
  if (areaRatio >= 0.02 || extent >= 0.3) return "critical";
  if (areaRatio >= 0.002 || extent >= 0.08) return "major";
  return crit >= 0.85 ? "major" : "minor";
}

function heuristicType(c: Component, gray: Float32Array, w: number, h: number): string {
  const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
  const long = Math.max(bw, bh), short = Math.max(1, Math.min(bw, bh));
  const fill = c.idx.length / (bw * bh);
  if (long / short >= 3.5 || (fill < 0.35 && long > 25)) return "linear_mark";
  const pad = Math.max(4, Math.round(0.5 * long));
  const inside = new Set<number>(c.idx);
  let si = 0, ni = 0, so = 0, no = 0;
  for (let y = Math.max(0, c.y0 - pad); y <= Math.min(h - 1, c.y1 + pad); y++)
    for (let x = Math.max(0, c.x0 - pad); x <= Math.min(w - 1, c.x1 + pad); x++) {
      const p = y * w + x;
      if (inside.has(p)) { si += gray[p]; ni++; } else { so += gray[p]; no++; }
    }
  const delta = ni && no ? si / ni - so / no : 0;
  if (c.idx.length / (w * h) > 0.01 && Math.abs(delta) < 25) return "blotch";
  return delta < 0 ? "dark_spot" : "bright_spot";
}

function makeFinding(c: Component, w: number, h: number, type: string, typeSource: Finding["typeSource"], typeConf: number | null,
  conf: number, basis: string, material: string): Finding {
  const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
  const t = config.taxonomy[type] ?? { label: type, category: "other", crit: 0.5 };
  return {
    type, label: t.label, category: t.category, typeSource, typeConf, conf: Math.min(1, Math.max(0, conf)), basis,
    box: [c.x0, c.y0, bw, bh], severity: severity(type, c.idx.length / (w * h), Math.max(bw, bh) / Math.max(w, h)),
    location: location(c.x0 + bw / 2, c.y0 + bh / 2, w, h),
    causes: typeSource === "model" ? config.causes[`${type}|${material}`] ?? [] : [],
  };
}

function paintMask(comps: Component[], w: number, h: number): Uint8Array {
  const m = new Uint8Array(w * h);
  comps.forEach((c, i) => c.idx.forEach((p) => (m[p] = i + 1)));
  return m;
}

/** Light version of the server's image-quality gate: an unusable image is never passed or rejected automatically. */
function quality(gray: Float32Array, w: number, h: number): { status: Result["quality"]; note: string | null } {
  if (Math.min(w, h) < 96) return { status: "fail", note: "image is too small to inspect" };
  let s = 0, s2 = 0, ext = 0;
  for (const v of gray) { s += v; s2 += v * v; if (v <= 40 || v >= 215) ext++; }
  const n = gray.length, mean = s / n, std = Math.sqrt(Math.max(0, s2 / n - mean * mean));
  if (std < 2) return { status: "fail", note: "almost no detail in the image (blank or covered lens)" };
  if (ext / n < 0.92 && (mean < 20 || mean > 238)) return { status: "fail", note: "image is severely under- or over-exposed" };
  let lap = 0, l2 = 0, m = 0;
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
      lap += v; l2 += v * v; m++;
    }
  const lv = l2 / m - (lap / m) ** 2;
  if (lv < 1.5) return { status: "fail", note: "image appears out of focus" };
  return { status: mean < 45 || mean > 215 || std < 6 ? "warn" : "ok", note: null };
}

async function embed(p: Pixels): Promise<Float32Array> {
  const r = await sFeat.run({ image: imageTensor(p) });
  return r.embeddings.data as Float32Array;
}

async function knn(emb: Float32Array, mem: ort.Tensor): Promise<Float32Array> {
  const r = await sKnn.run({ emb: new ort.Tensor("float32", emb, [emb.length / 128, 128]), mem });
  return r.dist.data as Float32Array;
}

// ------------------------------------------------------------------ mode 1: learn normal appearance from reference footage
export interface Reference { mem: ort.Tensor; w: number; h: number; med: number; mad: number; frames: number }
const K_SIGMA: Record<Sensitivity, number> = { low: 4.5, normal: 3.0, high: 2.0 };

export function learnSize(srcW: number, srcH: number, long = 448): [number, number] {
  const f = long / Math.max(srcW, srcH);
  const r32 = (v: number) => Math.max(96, Math.round((v * f) / 32) * 32);
  return [r32(srcW), r32(srcH)];
}

function smoothGrid(d: Float32Array, gw: number, gh: number, w: number, h: number): Float32Array {
  return gaussian(d, gw, gh, Math.max(0.6, (0.012 * Math.max(w, h)) / 8));
}

/** Greedy k-center coreset (as PatchCore) over a random pre-sample of patch embeddings. */
function coreset(all: Float32Array, n: number, presample: number, keep: number): Float32Array {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const sel = idx.slice(0, Math.min(presample, n));
  const m = sel.length;
  const pts = new Float32Array(m * 128);
  sel.forEach((s, i) => pts.set(all.subarray(s * 128, s * 128 + 128), i * 128));
  if (m <= keep) return pts;
  const dist = new Float32Array(m).fill(Infinity);
  const out = new Float32Array(keep * 128);
  let cur = 0;
  for (let k = 0; k < keep; k++) {
    out.set(pts.subarray(cur * 128, cur * 128 + 128), k * 128);
    let best = -1, bi = 0;
    const c0 = cur * 128;
    for (let i = 0; i < m; i++) {
      const o = i * 128;
      let d = 0;
      for (let j = 0; j < 128; j++) { const t = pts[o + j] - pts[c0 + j]; d += t * t; }
      if (d < dist[i]) dist[i] = d;
      if (dist[i] > best) { best = dist[i]; bi = i; }
    }
    cur = bi;
  }
  return out;
}

function sample(all: Float32Array, n: number, keep: number): Float32Array {
  if (n <= keep) return all.slice(0, n * 128);
  const out = new Float32Array(keep * 128);
  for (let k = 0; k < keep; k++) { const s = Math.floor(Math.random() * n); out.set(all.subarray(s * 128, s * 128 + 128), k * 128); }
  return out;
}

export async function learnReference(frames: CanvasImageSource[], srcW: number, srcH: number, onStep: (m: string) => void): Promise<Reference> {
  if (frames.length < 6) throw new Error("At least 6 defect-free reference frames are needed.");
  const [w, h] = learnSize(srcW, srcH);
  const gw = w / 8, gh = h / 8, P = gw * gh;
  const embs: Float32Array[] = [];
  for (let i = 0; i < frames.length; i++) {
    onStep(`Learning normal appearance: reading reference frame ${i + 1} of ${frames.length}`);
    embs.push(await embed(pixels(frames[i], w, h)));
  }
  const cat = (list: Float32Array[]) => { const a = new Float32Array(list.length * P * 128); list.forEach((e, i) => a.set(e, i * P * 128)); return a; };
  // threshold: how anomalous does a normal frame look against the other normal frames (two-fold)?
  onStep("Learning normal appearance: calibrating the threshold");
  const peaks: number[] = [];
  for (const fold of [0, 1]) {
    const fitSet = embs.filter((_, i) => i % 2 !== fold), held = embs.filter((_, i) => i % 2 === fold);
    const mem = new ort.Tensor("float32", sample(cat(fitSet), fitSet.length * P, 6000), [Math.min(6000, fitSet.length * P), 128]);
    for (const e of held) {
      const g = smoothGrid(await knn(e, mem), gw, gh, w, h);
      peaks.push(Math.max(...g));
    }
  }
  onStep("Learning normal appearance: building the memory of normal patches");
  await new Promise((r) => setTimeout(r, 30));
  const core = coreset(cat(embs), embs.length * P, 12000, 2500);
  const med = median(peaks);
  const mad = Math.max(median(peaks.map((p) => Math.abs(p - med))) * 1.4826, 0.02 * med);
  return { mem: new ort.Tensor("float32", core, [core.length / 128, 128]), w, h, med, mad, frames: frames.length };
}

function median(a: number[]) { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

export async function inspectLearned(src: CanvasImageSource, ref: Reference, sens: Sensitivity, material: string): Promise<Result> {
  const t0 = performance.now();
  const { w, h } = ref;
  const gw = w / 8, gh = h / 8;
  const px = pixels(src, w, h);
  const q = quality(px.gray, w, h);
  const thr = ref.med + K_SIGMA[sens] * ref.mad;
  const full = resize(smoothGrid(await knn(await embed(px), ref.mem), gw, gh, w, h), gw, gh, w, h);
  const comps = components(full, w, h, thr, Math.max(16, Math.round(0.0004 * w * h)));
  const findings = comps.map((c) => makeFinding(c, w, h, heuristicType(c, px.gray, w, h), "heuristic", null, c.peak / (1.6 * thr), "deviation from learned normal", material));
  const reasons = findings.length
    ? [`${findings.length} area(s) differ from the learned normal appearance. A person should confirm whether they are defects.`]
    : ["No area differs from the learned normal appearance."];
  reasons.push("This detector was learned from your footage and has no measured accuracy, so it never passes or rejects a part by itself.");
  let outcome: Outcome = findings.length ? "review" : "clear";
  if (q.status === "fail") { outcome = "review"; reasons.unshift(`Image quality: ${q.note}.`); }
  return { outcome, pDefect: null, reasons, findings, mask: paintMask(comps, w, h), maskW: w, maskH: h, ms: performance.now() - t0, quality: q.status };
}

// ------------------------------------------------------------------ mode 2: trained pavement / concrete crack model
export async function inspectCrack(src: CanvasImageSource, material: string): Promise<Result> {
  const t0 = performance.now();
  const c = config.crack;
  const [h, w] = c.segInput;
  const px = pixels(src, w, h);
  const q = quality(px.gray, w, h);
  const out = (await (await session("crack_seg")).run({ image: imageTensor(px) })).probs.data as Float32Array;
  const n = w * h;
  const defect = new Float32Array(n);
  for (let i = 0; i < n; i++) defect[i] = 1 - out[i];
  const comps = components(defect, w, h, c.segPixelThreshold, Math.max(6, Math.round(0.00005 * n)));
  const findings = comps.map((cc) => makeFinding(cc, w, h, "crack", "model", cc.peak, cc.peak, "model probability", material));
  const strong = findings.filter((f) => f.conf >= c.regionConfident);
  let outcome: Outcome, reasons: string[];
  if (strong.length) {
    outcome = "defective";
    reasons = [`${strong.length} crack region(s) with model confidence of at least ${Math.round(c.regionConfident * 100)}%. Crack localization was validated on held-out CrackForest images.`];
  } else if (findings.length) {
    outcome = "review";
    reasons = [`Possible cracks found, but none reaches the ${Math.round(c.regionConfident * 100)}% confidence needed to reject automatically.`];
  } else {
    outcome = "clear";
    reasons = ["No cracks found. This model was never validated on crack-free surfaces, so this is not a certified pass."];
  }
  if (q.status === "fail") { outcome = "review"; reasons.unshift(`Image quality: ${q.note}.`); }
  return { outcome, pDefect: null, reasons, findings, mask: paintMask(comps, w, h), maskW: w, maskH: h, ms: performance.now() - t0, quality: q.status };
}

// ------------------------------------------------------------------ mode 3: trained ferrite magnetic tile model (two detectors + calibrated fusion)
export interface TileScores { pc: number; seg: number; p: number }

export async function inspectTile(src: CanvasImageSource, material: string): Promise<Result & { scores: TileScores }> {
  const t0 = performance.now();
  const c = config.tile;
  const [h, w] = c.segInput;
  const px = pixels(src, w, h);
  const q = quality(px.gray, w, h);
  const probs = (await (await session("tile_seg")).run({ image: imageTensor(px) })).probs.data as Float32Array;
  const n = w * h, K = c.classes.length;
  const defect = new Float32Array(n);
  for (let i = 0; i < n; i++) defect[i] = 1 - probs[i];
  const sorted = Float32Array.from(defect).sort();
  const k = Math.max(8, Math.floor(n / 1000));
  let seg = 0;
  for (let i = n - k; i < n; i++) seg += sorted[i];
  seg /= k;
  // PatchCore anomaly score on its own input size
  const [ph, pw] = c.pcInput;
  const d = await knn(await embed(pixels(src, pw, ph)), await tileMem());
  const pc = Math.max(...gaussian(d, pw / 8, ph / 8, 1.0));
  const z = c.fusion.coef[0] * pc + c.fusion.coef[1] * seg + c.fusion.intercept;
  const p = 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, z))));
  const comps = components(defect, w, h, c.segPixelThreshold, Math.max(6, Math.round(0.00005 * n)));
  const findings = comps.map((cc) => {
    const votes = new Float32Array(K), mean = new Float32Array(K);
    let dsum = 0;
    for (const pix of cc.idx) {
      let best = 0;
      for (let j = 1; j < K; j++) if (probs[(1 + j) * n + pix] > probs[(1 + best) * n + pix]) best = j;
      votes[best]++;
      for (let j = 0; j < K; j++) mean[j] += probs[(1 + j) * n + pix];
      dsum += defect[pix];
    }
    const kk = votes.indexOf(Math.max(...votes));
    return makeFinding(cc, w, h, c.classes[kk], "model", Math.min(1, mean[kk] / Math.max(dsum, 1e-6)), cc.peak, "model probability", material);
  });
  const pl = (name: string, s: number) => 1 / (1 + Math.exp(-(c.platt[name].a * s + c.platt[name].b)));
  const ppc = pl("mt_patchcore", pc), pseg = pl("mt_segnet", seg);
  let outcome: Outcome, reasons: string[];
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  if (p <= c.fusion.t_low) { outcome = "pass"; reasons = [`Calibrated defect probability ${pct(p)} is below the validated pass threshold ${pct(c.fusion.t_low)}.`]; }
  else if (p >= c.fusion.t_high) { outcome = "defective"; reasons = [`Calibrated defect probability ${pct(p)} is above the validated reject threshold ${pct(c.fusion.t_high)}.`]; }
  else { outcome = "review"; reasons = [`Calibrated defect probability ${pct(p)} lies inside the review band (${pct(c.fusion.t_low)} to ${pct(c.fusion.t_high)}).`]; }
  if (outcome !== "review" && Math.abs(ppc - pseg) >= c.disagreementDelta) {
    outcome = "review";
    reasons.push(`The two detectors disagree strongly (anomaly model ${pct(ppc)}, segmentation model ${pct(pseg)}).`);
  }
  if (q.status === "fail") { outcome = "review"; reasons.unshift(`Image quality: ${q.note}.`); }
  return { outcome, pDefect: p, reasons, findings, mask: paintMask(comps, w, h), maskW: w, maskH: h, ms: performance.now() - t0, quality: q.status, scores: { pc, seg, p } };
}
