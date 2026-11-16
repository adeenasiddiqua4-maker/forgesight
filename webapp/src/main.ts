import "./style.css";
import * as E from "./engine";

type Src = { kind: "video" | "images" | "camera" | "sample"; name: string };
interface Entry { at: string; source: string; outcome: E.Outcome; n: number; top: string; maxConf: number; p: number | null; snap: string; reasons: string[] }

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const pct = (v: number | null, d = 0) => (v === null ? "–" : `${(v * 100).toFixed(d)}%`);
const fmtT = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const OUT: Record<E.Outcome, { text: string; cls: string; label: string }> = {
  pass: { text: "PASS", cls: "ok", label: "Pass" },
  clear: { text: "CLEAR*", cls: "clear", label: "Clear*" },
  review: { text: "CHECK", cls: "warn", label: "Check" },
  defective: { text: "NG", cls: "ng", label: "Defective" },
};
const SEV: Record<string, string> = { minor: "#e0a325", major: "#ec7f45", critical: "#e0463e" };

document.querySelector("#app")!.innerHTML = `
<header class="top">
  <div class="brand"><span class="mark" aria-hidden="true"></span><div><div class="name">ForgeSight</div><div class="sub">Visual defect inspection</div></div></div>
  <div class="status" id="status"><span class="dot"></span><span id="statusText">Loading models…</span></div>
</header>
<p class="lead">Load a video, photos or a camera feed of the material. Each analysed frame is shown with the defects marked. Everything runs inside this browser: your footage is never uploaded.</p>
<main class="grid">
  <section class="viewer-col">
    <div class="panel viewer">
      <div class="stage" id="stage">
        <canvas id="view" width="960" height="540" aria-label="Analysed frame"></canvas>
        <div class="empty" id="empty">
          <div class="drop" id="drop">
            <strong>Drop a video or photos here</strong>
            <span>MP4 (H.264) or WebM video · JPEG / PNG photos</span>
          </div>
          <div class="samples"><span>Or try a sample</span>
            <button class="btn ghost" data-sample="samples/pavement_line_video.webm" data-mode="crack">Concrete cracks</button>
            <button class="btn ghost" data-sample="samples/tile_line_video.webm" data-mode="learn" data-ref="6">Ferrite tiles, learned</button>
          </div>
        </div>
        <div class="phase" id="phase" hidden></div>
      </div>
      <div class="bar"><div class="fill" id="progress"></div></div>
      <div class="controls">
        <label class="btn primary" id="btnVideo">Analyse video<input id="inVideo" type="file" accept="video/*" hidden></label>
        <label class="btn" id="btnImages">Analyse photos<input id="inImages" type="file" accept="image/*" multiple hidden></label>
        <button class="btn" id="btnCam">Camera</button>
        <button class="btn danger" id="btnStop" hidden>Stop</button>
        <span class="hint" id="hint"></span>
      </div>
    </div>
    <div class="panel" id="framePanel">
      <h2>Current frame</h2>
      <div id="frameInfo" class="muted">Nothing analysed yet.</div>
    </div>
  </section>

  <aside class="side-col">
    <div class="panel">
      <h2>Setup</h2>
      <label class="field"><span>Material</span><select id="material"></select></label>
      <label class="field"><span>Detection</span>
        <select id="mode">
          <option value="learn">Learn normal from this footage (any material)</option>
          <option value="crack">Trained model: pavement / concrete cracks</option>
          <option value="tile">Trained model: ferrite magnetic tiles</option>
        </select>
      </label>
      <p class="note" id="modeNote"></p>
      <div id="learnOpts">
        <div class="row2">
          <label class="field"><span id="refLabel">Defect-free seconds at start</span><input id="refSec" inputmode="decimal" value="5"></label>
          <label class="field"><span>Sensitivity</span><select id="sens"><option value="low">Low</option><option value="normal" selected>Normal</option><option value="high">High</option></select></label>
        </div>
      </div>
      <label class="field"><span>Analyse every (seconds of video)</span><input id="step" inputmode="decimal" value="0.5"></label>
      <div class="field"><span>Sample videos</span>
        <div class="samplebtns">
          <button class="btn small" data-sample="samples/pavement_line_video.webm" data-mode="crack">Concrete cracks</button>
          <button class="btn small" data-sample="samples/tile_line_video.webm" data-mode="learn" data-ref="6">Ferrite tiles, learned</button>
        </div>
      </div>
    </div>
    <div class="counts">
      <div class="count"><span>Frames</span><b id="cFrames">0</b><em id="cMs">–</em></div>
      <div class="count ok"><span>OK</span><b id="cOk">0</b></div>
      <div class="count warn"><span>Check</span><b id="cCheck">0</b></div>
      <div class="count ng"><span>NG</span><b id="cNg">0</b></div>
    </div>
    <div class="panel">
      <div class="h2row"><h2>Flagged frames <span id="nFlag" class="muted">0</span></h2><button class="btn small" id="btnCopy" disabled>Copy results</button></div>
      <ul class="flags" id="flags"><li class="muted">Frames that need attention appear here.</li></ul>
    </div>
  </aside>
</main>
<section class="panel facts">
  <h2>How reliable is it?</h2>
  <div class="facts-grid">
    <div><h3>Trained models</h3><p>Measured on held-out public test images the models never saw. <b>Concrete cracks</b> (CrackForest, 30 images): crack pixel F1 0.65, or 0.87 allowing 2 px. <b>Ferrite tiles</b> (277 images): AUROC 0.998, no missed defects and 2 false alarms; about 4 in 10 parts are sent to a person for review instead of being decided automatically.</p></div>
    <div><h3>Learn from footage</h3><p>Works on any material, leather included, because it needs no trained model. It learns what normal looks like from the first seconds and marks anything different. On a hard test (tile stills, a different part every frame) it marked 49% / 80% / 89% of defective frames with 12% / 33% / 51% false marks for low / normal / high sensitivity. It is a screening aid: a person confirms each mark.</p></div>
    <div><h3>Good footage</h3><p>Keep the camera, distance and lighting fixed. The defect-free seconds at the start must contain only good material. Raking light shows scratches and wrinkles on leather; diffuse light shows stains. *CLEAR means nothing was found, not a certified pass.</p></div>
  </div>
</section>`;

const cv = $("#view") as HTMLCanvasElement;
const ctx = cv.getContext("2d")!;
let running = false;
let stopFlag = false;
let ref: E.Reference | null = null;
let entries: Entry[] = [];
let counts = { frames: 0, ok: 0, check: 0, ng: 0, ms: 0 };

function setStatus(text: string, cls = "") { $("#statusText").textContent = text; $("#status").className = `status ${cls}`; }
function phase(msg: string | null) { const p = $("#phase"); p.hidden = !msg; p.textContent = msg ?? ""; }
function progress(v: number | null) { ($("#progress") as HTMLElement).style.width = v === null ? "0" : `${Math.round(v * 100)}%`; }

function modeNote() {
  const m = ($("#mode") as HTMLSelectElement).value as E.Mode;
  const mat = $("#material") as HTMLSelectElement;
  $("#learnOpts").hidden = m !== "learn";
  if (m === "crack") mat.value = "construction";
  if (m === "tile") mat.value = "ceramics";
  mat.disabled = m !== "learn";
  $("#modeNote").textContent = {
    learn: "Learns what good material looks like from the start of your video (or the first photos), then marks anything that differs. Use this for leather and any material without a trained model.",
    crack: "Model trained on pavement and concrete crack photos. Finds and outlines cracks.",
    tile: "Two models trained on ferrite magnetic tiles, combined into a calibrated pass / check / reject decision. Recognises blowholes, breaks, cracks, fray and uneven surface.",
  }[m];
}

function resetRun() {
  entries = [];
  counts = { frames: 0, ok: 0, check: 0, ng: 0, ms: 0 };
  ref = null;
  renderCounts();
  renderFlags();
  $("#frameInfo").innerHTML = '<span class="muted">Nothing analysed yet.</span>';
}

function renderCounts() {
  $("#cFrames").textContent = String(counts.frames);
  $("#cOk").textContent = String(counts.ok);
  $("#cCheck").textContent = String(counts.check);
  $("#cNg").textContent = String(counts.ng);
  $("#cMs").textContent = counts.frames ? `${Math.round(counts.ms / counts.frames)} ms / frame` : "–";
}

function renderFlags() {
  const ul = $("#flags");
  const flagged = entries.filter((e) => e.outcome === "review" || e.outcome === "defective");
  $("#nFlag").textContent = String(flagged.length);
  ($("#btnCopy") as HTMLButtonElement).disabled = entries.length === 0;
  if (!flagged.length) { ul.innerHTML = `<li class="muted">${entries.length ? "No frame needed attention." : "Frames that need attention appear here."}</li>`; return; }
  ul.innerHTML = flagged.slice().reverse().slice(0, 200).map((e) =>
    `<li><button data-i="${entries.indexOf(e)}"><span class="t">${e.at}</span><span class="badge ${OUT[e.outcome].cls}">${OUT[e.outcome].text}</span><span class="what">${e.n} · ${e.top}</span></button></li>`).join("");
}

$("#flags").addEventListener("click", (ev) => {
  const b = (ev.target as HTMLElement).closest("button[data-i]") as HTMLButtonElement | null;
  if (!b || running) return;
  const e = entries[+b.dataset.i!];
  const img = new Image();
  img.onload = () => { cv.width = img.width; cv.height = img.height; ctx.drawImage(img, 0, 0); };
  img.src = e.snap;
  $("#frameInfo").innerHTML = `<p class="muted">Stored frame ${e.at} (${e.source}).</p><ul class="reasons">${e.reasons.map((r) => `<li>${r}</li>`).join("")}</ul>`;
});

$("#btnCopy").addEventListener("click", () => {
  const rows = [["source", "position", "outcome", "findings", "top_finding", "max_confidence", "defect_probability"],
    ...entries.map((e) => [e.source, e.at, OUT[e.outcome].label, e.n, e.top, e.maxConf.toFixed(3), e.p === null ? "" : e.p.toFixed(4)])];
  const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const btn = $("#btnCopy");
  navigator.clipboard.writeText(csv).then(() => { btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy results"), 1500); },
    () => { btn.textContent = "Copy blocked"; setTimeout(() => (btn.textContent = "Copy results"), 1500); });
});

// ------------------------------------------------------------------ drawing
function draw(src: CanvasImageSource, sw: number, sh: number, r: E.Result, stamp: string, mode: E.Mode) {
  const f = Math.min(1, 960 / Math.max(sw, sh));
  cv.width = Math.round(sw * f);
  cv.height = Math.round(sh * f);
  ctx.drawImage(src, 0, 0, cv.width, cv.height);
  // masks
  const mc = document.createElement("canvas");
  mc.width = r.maskW; mc.height = r.maskH;
  const mctx = mc.getContext("2d")!;
  const id = mctx.createImageData(r.maskW, r.maskH);
  const rgb = r.findings.map((x) => SEV[x.severity]).map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  for (let i = 0; i < r.mask.length; i++) {
    const k = r.mask[i];
    if (!k) continue;
    const c = rgb[k - 1];
    id.data[4 * i] = c[0]; id.data[4 * i + 1] = c[1]; id.data[4 * i + 2] = c[2]; id.data[4 * i + 3] = 105;
  }
  mctx.putImageData(id, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(mc, 0, 0, cv.width, cv.height);
  // boxes + labels
  const sx = cv.width / r.maskW, sy = cv.height / r.maskH;
  const u = Math.max(cv.width, cv.height) / 640;
  ctx.font = `700 ${Math.round(11 * u)}px "IBM Plex Mono", ui-monospace, monospace`;
  r.findings.forEach((x) => {
    const col = SEV[x.severity];
    const [bx, by, bw, bh] = [x.box[0] * sx, x.box[1] * sy, x.box[2] * sx, x.box[3] * sy];
    ctx.strokeStyle = col; ctx.lineWidth = 2 * u; ctx.strokeRect(bx, by, bw, bh);
    const text = `${(x.typeSource === "model" ? x.label : x.label.split(" (")[0]).toUpperCase()} ${Math.round(x.conf * 100)}%`;
    const tw = ctx.measureText(text).width + 8 * u, th = 15 * u;
    const ty = by - th - 2 * u > 32 * u ? by - th - 2 * u : by + bh + 2 * u;
    const tx = Math.min(Math.max(0, bx), cv.width - tw);
    ctx.fillStyle = col; ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = "#111"; ctx.fillText(text, tx + 4 * u, ty + 11 * u);
  });
  // status bar
  const o = OUT[r.outcome];
  const barH = 30 * u;
  ctx.fillStyle = "rgba(0,0,0,0.74)"; ctx.fillRect(0, 0, cv.width, barH);
  ctx.fillStyle = "#f2f1ec"; ctx.font = `600 ${Math.round(10 * u)}px "IBM Plex Mono", ui-monospace, monospace`;
  const modeName = { learn: "LEARNED REFERENCE", crack: "CONCRETE CRACK MODEL", tile: "FERRITE TILE MODEL" }[mode];
  ctx.fillText(`MODE ${modeName}   ${stamp}   ${Math.round(r.ms)} ms`, 6 * u, 12 * u);
  ctx.fillText(`FRAMES ${counts.frames}   OK ${counts.ok}   CHECK ${counts.check}   NG ${counts.ng}`, 6 * u, 24 * u);
  const bc = { ok: "#2fa84f", clear: "#9aa0a6", warn: "#e0a325", ng: "#e0463e" }[o.cls]!;
  ctx.fillStyle = bc; ctx.fillRect(cv.width - 84 * u, 4 * u, 78 * u, 22 * u);
  ctx.fillStyle = "#111"; ctx.font = `800 ${Math.round(13 * u)}px "IBM Plex Mono", ui-monospace, monospace`;
  ctx.textAlign = "center"; ctx.fillText(o.text, cv.width - 45 * u, 20 * u); ctx.textAlign = "start";
  ctx.strokeStyle = bc; ctx.lineWidth = 3 * u; ctx.strokeRect(1.5 * u, 1.5 * u, cv.width - 3 * u, cv.height - 3 * u);
}

function frameInfo(r: E.Result) {
  const rows = r.findings.slice(0, 10).map((f) => `<tr><td>${f.label}${f.typeSource === "model" ? "" : '<br><small>type not established</small>'}</td><td>${f.location}</td><td class="num">${pct(f.conf)}</td><td><span class="sev" style="--c:${SEV[f.severity]}">${f.severity}</span></td></tr>`).join("");
  const causes = [...new Set(r.findings.flatMap((f) => f.causes))].slice(0, 5);
  $("#frameInfo").innerHTML = `
    <div class="verdict"><span class="badge big ${OUT[r.outcome].cls}">${OUT[r.outcome].text}</span>${r.pDefect !== null ? `<span class="muted">defect probability <b class="num">${pct(r.pDefect, 1)}</b></span>` : ""}</div>
    <ul class="reasons">${r.reasons.map((x) => `<li>${x}</li>`).join("")}</ul>
    ${rows ? `<div class="tablewrap"><table><thead><tr><th>Finding</th><th>Where</th><th>Confidence</th><th>Severity</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
    ${causes.length ? `<div class="hyp"><h3>Possible causes (hypotheses, not observed)</h3><ul>${causes.map((c) => `<li>${c}</li>`).join("")}</ul></div>` : ""}`;
}

function record(r: E.Result, stamp: string, source: string) {
  counts.frames++; counts.ms += r.ms;
  if (r.outcome === "pass" || r.outcome === "clear") counts.ok++;
  else if (r.outcome === "review") counts.check++;
  else counts.ng++;
  const flagged = r.outcome === "review" || r.outcome === "defective";
  entries.push({ at: stamp, source, outcome: r.outcome, n: r.findings.length, top: r.findings[0]?.label ?? "none",
    maxConf: Math.max(0, ...r.findings.map((f) => f.conf)), p: r.pDefect, snap: flagged ? cv.toDataURL("image/jpeg", 0.82) : "", reasons: r.reasons });
  renderCounts();
  if (flagged) renderFlags(); else ($("#btnCopy") as HTMLButtonElement).disabled = false;
}

// ------------------------------------------------------------------ running
const getMode = () => ($("#mode") as HTMLSelectElement).value as E.Mode;
const getMat = () => ($("#material") as HTMLSelectElement).value;
const num = (sel: string, d: number) => { const v = parseFloat(($(sel) as HTMLInputElement).value); return Number.isFinite(v) && v > 0 ? v : d; };

async function analyse(src: CanvasImageSource): Promise<E.Result> {
  const m = getMode();
  if (m === "crack") return E.inspectCrack(src, getMat());
  if (m === "tile") return E.inspectTile(src, getMat());
  return E.inspectLearned(src, ref!, ($("#sens") as HTMLSelectElement).value as E.Sensitivity, getMat());
}

function begin() {
  running = true; stopFlag = false;
  $("#empty").hidden = true;
  ["#btnVideo", "#btnImages", "#btnCam"].forEach((s) => $(s).classList.add("disabled"));
  ["#mode", "#material", "#refSec", "#sens", "#step"].forEach((s) => (($(s) as HTMLInputElement).disabled = true));
  $("#btnStop").hidden = false;
  $("#hint").textContent = "";
  resetRun();
}
function end(msg?: string) {
  running = false;
  ["#btnVideo", "#btnImages", "#btnCam"].forEach((s) => $(s).classList.remove("disabled"));
  ["#mode", "#refSec", "#sens", "#step"].forEach((s) => (($(s) as HTMLInputElement).disabled = false));
  modeNote();
  $("#btnStop").hidden = true;
  phase(null);
  $("#hint").textContent = msg ?? "";
  renderFlags();
}

function fail(e: unknown) {
  console.error(e);
  end();
  $("#hint").textContent = `Stopped: ${(e as Error).message ?? e}`;
}

async function runVideo(srcUrl: string, name: string) {
  begin();
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = srcUrl;
  try {
    await new Promise<void>((res, rej) => { v.onloadeddata = () => res(); v.onerror = () => rej(new Error("this video format cannot be decoded by the browser. Use MP4 (H.264) or WebM.")); });
  } catch (e) { return fail(e); }
  const seek = (t: number) => new Promise<void>((res) => { v.onseeked = () => res(); v.currentTime = t; });
  const step = num("#step", 0.5);
  let start = 0;
  try {
    if (getMode() === "learn") {
      const span = Math.min(num("#refSec", 5), v.duration * 0.6);
      if (span < 0.4) throw new Error("the video is too short to learn a reference from.");
      const frames: HTMLCanvasElement[] = [];
      for (let i = 0; i < 12 && !stopFlag; i++) {
        phase(`Reading defect-free reference frame ${i + 1} of 12 from the first ${span.toFixed(1)} s`);
        await seek((i / 12) * span);
        const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d")!.drawImage(v, 0, 0); frames.push(c);
      }
      if (stopFlag) return end("Stopped.");
      ref = await E.learnReference(frames, v.videoWidth, v.videoHeight, phase);
      start = span;
    }
    phase(null);
    for (let t = start; t < v.duration && !stopFlag; t += step) {
      await seek(t);
      const r = await analyse(v);
      const stamp = `t=${fmtT(t)}`;
      draw(v, v.videoWidth, v.videoHeight, r, stamp, getMode());
      record(r, stamp, name);
      draw(v, v.videoWidth, v.videoHeight, r, stamp, getMode()); // redraw so the status bar shows updated counts
      frameInfo(r);
      progress(Math.min(1, (t + step) / v.duration));
      await new Promise((res) => setTimeout(res, 120));
    }
    end(stopFlag ? "Stopped." : `Finished: ${counts.frames} frames analysed.`);
  } catch (e) { fail(e); }
}

async function runImages(files: File[]) {
  begin();
  const imgs = files.filter((f) => f.type.startsWith("image/")).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const load = (f: File) => createImageBitmap(f);
  try {
    let from = 0;
    if (getMode() === "learn") {
      const nRef = Math.round(num("#refSec", 6));
      if (imgs.length < nRef + 1 || nRef < 6) throw new Error(`learning needs at least 6 defect-free photos first plus photos to check (set to ${nRef}, got ${imgs.length} photos).`);
      const first = await load(imgs[0]);
      const frames: ImageBitmap[] = [first];
      for (let i = 1; i < nRef; i++) frames.push(await load(imgs[i]));
      ref = await E.learnReference(frames, first.width, first.height, phase);
      from = nRef;
    }
    phase(null);
    for (let i = from; i < imgs.length && !stopFlag; i++) {
      const bmp = await load(imgs[i]);
      const r = await analyse(bmp);
      record(r, imgs[i].name, "photo");
      draw(bmp, bmp.width, bmp.height, r, imgs[i].name, getMode());
      frameInfo(r);
      progress((i + 1) / imgs.length);
      await new Promise((res) => setTimeout(res, 60));
    }
    end(stopFlag ? "Stopped." : `Finished: ${counts.frames} photos analysed.`);
  } catch (e) { fail(e); }
}

async function runCamera() {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }, audio: false });
  } catch {
    $("#hint").textContent = "The camera is not available here (permission refused or no camera). Record a video and use Analyse video instead.";
    return;
  }
  begin();
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.srcObject = stream;
  await v.play();
  const grab = () => { const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d")!.drawImage(v, 0, 0); return c; };
  try {
    if (getMode() === "learn") {
      const span = num("#refSec", 5);
      const frames: HTMLCanvasElement[] = [];
      for (let i = 0; i < 12 && !stopFlag; i++) {
        phase(`Show defect-free material: capturing reference ${i + 1} of 12`);
        await new Promise((r) => setTimeout(r, (span * 1000) / 12));
        frames.push(grab());
      }
      if (!stopFlag) ref = await E.learnReference(frames, v.videoWidth, v.videoHeight, phase);
    }
    phase(null);
    const t0 = performance.now();
    while (!stopFlag) {
      const c = grab();
      const r = await analyse(c);
      const stamp = `live ${fmtT((performance.now() - t0) / 1000)}`;
      record(r, stamp, "camera");
      draw(c, c.width, c.height, r, stamp, getMode());
      frameInfo(r);
      await new Promise((res) => setTimeout(res, Math.max(50, num("#step", 0.5) * 1000 - r.ms)));
    }
    end("Camera stopped.");
  } catch (e) { fail(e); }
  stream.getTracks().forEach((t) => t.stop());
}

// ------------------------------------------------------------------ wiring
$("#mode").addEventListener("change", () => {
  const m = getMode();
  $("#refLabel").textContent = "Defect-free seconds at start";
  if (m !== "learn") return modeNote();
  modeNote();
});
$("#inVideo").addEventListener("change", (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f && !running) runVideo(URL.createObjectURL(f), f.name); (e.target as HTMLInputElement).value = ""; });
$("#inImages").addEventListener("change", (e) => {
  const fs = Array.from((e.target as HTMLInputElement).files ?? []);
  if (fs.length && !running) {
    if (getMode() === "learn" && ($("#refSec") as HTMLInputElement).value === "5") ($("#refSec") as HTMLInputElement).value = "6";
    runImages(fs);
  }
  (e.target as HTMLInputElement).value = "";
});
$("#btnCam").addEventListener("click", () => { if (!running) runCamera(); });
$("#btnStop").addEventListener("click", () => { stopFlag = true; });
document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((b) => b.addEventListener("click", () => {
  if (running) return;
  ($("#mode") as HTMLSelectElement).value = b.dataset.mode!;
  if (b.dataset.mode === "learn") { ($("#material") as HTMLSelectElement).value = "ceramics"; ($("#refSec") as HTMLInputElement).value = b.dataset.ref!; }
  modeNote();
  runVideo(new URL(b.dataset.sample!, document.baseURI).href, b.textContent!.trim());
}));
const stage = $("#stage");
["dragenter", "dragover"].forEach((t) => stage.addEventListener(t, (e) => { e.preventDefault(); $("#drop").classList.add("over"); }));
["dragleave", "drop"].forEach((t) => stage.addEventListener(t, () => $("#drop").classList.remove("over")));
stage.addEventListener("drop", (e) => {
  e.preventDefault();
  if (running) return;
  const fs = Array.from((e as DragEvent).dataTransfer?.files ?? []);
  const vid = fs.find((f) => f.type.startsWith("video/"));
  if (vid) runVideo(URL.createObjectURL(vid), vid.name);
  else if (fs.length) runImages(fs);
});

const lock = (on: boolean) => ["#btnVideo", "#btnImages", "#btnCam"].concat(Array.from(document.querySelectorAll("[data-sample]")).map(() => "")).forEach((s) => s && $(s).classList.toggle("disabled", on));
lock(true);
document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((b) => (b.disabled = true));
E.loadEngine((m) => setStatus(`${m}…`)).then(() => {
  const sel = $("#material") as HTMLSelectElement;
  sel.innerHTML = E.config.materials.map((m) => `<option value="${m.key}">${m.name}</option>`).join("");
  sel.value = "leather";
  modeNote();
  setStatus("Ready · runs in your browser", "ready");
  lock(false);
  document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((b) => (b.disabled = false));
}).catch((e) => {
  console.error(e);
  setStatus("Could not start the detector", "bad");
  $("#hint").textContent = `This browser could not load the detection engine (${(e as Error).message}). Try a recent Chrome, Edge or Firefox.`;
});

// automated accuracy check hook (used by the test suite; harmless for viewers)
(window as unknown as Record<string, unknown>).__fs = { E, setMode: (m: E.Mode) => { ($("#mode") as HTMLSelectElement).value = m; modeNote(); } };
