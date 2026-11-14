/* Station twin dashboard (live). Reads only from the dashboard server:
 *   /api/config  camera poses + intrinsics (from the sim scene), pivot
 *   /api/status  source health, focus block and anomaly overlays (stubs until a service posts)
 *   /api/cloud   fused D415 point cloud (int16 x5 records), long-polled
 *   /frame/{sim,real}.jpg?after=SEQ  live Top camera frames, pulled one at a time so the rate
 *                     fits the link (the hidden source refreshes once a second)
 */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (el, name, fallback) => getComputedStyle(el).getPropertyValue(name).trim() || fallback;
  const hex = (h) => { const v = parseInt(h.replace("#", ""), 16); return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255]; };

  // fetch with a deadline: a half-dead link must not stall a loop forever
  function fetchT(url, ms) {
    const ac = new AbortController(), t = setTimeout(() => ac.abort(), ms);
    return fetch(url, { cache: "no-store", signal: ac.signal }).finally(() => clearTimeout(t));
  }
  const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  const state = {
    src: hash.src === "real" ? "real" : "sim",
    defects: hash.defects === "1",
    focus: hash.focus !== "0",
    scope: hash.scope === "focus" ? "focus" : "all",
    fs: ["cam", "depth"].includes(hash.fs) ? hash.fs : "",
    color: hash.color === "sensor" ? "sensor" : "rgb",
    zoom: hash.zoom === "2",
  };
  let config = null; // from /api/config
  let status = null; // from /api/status
  let cloud = null; // renderer

  // ------------------------------------------------------------------ state → DOM
  function bind(name, text) { $$(`[data-bind="${name}"]`).forEach((e) => (e.textContent = text)); }
  function setOn(el, on) { el.classList.toggle("on", !!on); el.setAttribute("aria-pressed", on ? "true" : "false"); }
  // A status older than STATUS_STALE_MS (local clock) means the server is unreachable: no source is
  // trusted and no overlay is drawn. Overlays must never outlive their data.
  const STATUS_STALE_MS = 4000;
  let statusAt = 0;
  const statusFresh = () => !!status && performance.now() - statusAt < STATUS_STALE_MS;
  const focusOverlay = () => (statusFresh() && status.focus?.connected && status.focus.object_id ? status.focus : null);
  function visibleAnomalies() {
    const a = status?.anomalies;
    if (!state.defects || !statusFresh() || !a?.connected) return [];
    const f = focusOverlay();
    return state.scope === "all" ? a.items : a.items.filter((i) => f && i.object_id === f.object_id);
  }

  function sync() {
    if (typeof wake === "function") wake();
    const b = document.body.classList;
    b.toggle("src-sim", state.src === "sim");
    b.toggle("src-real", state.src === "real");
    b.toggle("opt-defects", state.defects);
    b.toggle("opt-focus", state.focus);
    b.toggle("fs-cam", state.fs === "cam");
    b.toggle("fs-depth", state.fs === "depth");
    $$("[data-src]").forEach((e) => setOn(e, e.dataset.src === state.src));
    $$("[data-opt]").forEach((e) => setOn(e, state[e.dataset.opt]));
    $$("[data-scope]").forEach((e) => setOn(e, e.dataset.scope === state.scope));
    $$("[data-color]").forEach((e) => setOn(e, e.dataset.color === state.color));
    $$("[data-zoom]").forEach((e) => setOn(e, state.zoom));
    $$("[data-panel]").forEach((p) => p.classList.toggle("fs", p.dataset.panel === state.fs));
    $$("[data-fs]").forEach((e) => setOn(e, e.closest("[data-panel]")?.dataset.panel === state.fs));
    bind("zoom", state.zoom ? "2×" : "1×");
    bind("src", state.src === "sim" ? "SIM" : "REAL");
    bind("src-long", state.src === "sim" ? "MuJoCo sim" : "Real camera");
    renderStatus();
    drawOverlay();
    cloud?.resize();
  }

  function fmtAge(s) { return s == null ? "–" : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`; }

  function renderStatus() {
    if (!status) return;
    const fresh = statusFresh(), S = status.sources, b = document.body.classList;
    b.toggle("real-down", !fresh || !S.real.ok);
    b.toggle("sim-down", !fresh || !S.sim.ok);
    b.toggle("cloud-down", !fresh || !S.cloud.ok);
    const live = $('[data-bind="health"]');
    live.textContent = !fresh ? "Server unreachable" : status.ok ? "Live" : "Degraded";
    live.classList.toggle("down", !fresh || !status.ok);
    if (!fresh) {
      bind("cam-error", "The dashboard server does not answer.");
      bind("cloud-error", "The dashboard server does not answer.");
      bind("focus-label", "–");
      bind("defect-count", "0");
      for (const kind of ["anomalies", "focus"]) {
        $(`[data-stub="${kind}"]`).classList.toggle("show", state[kind === "anomalies" ? "defects" : "focus"]);
        bind(`${kind}-reason`, "Server unreachable");
      }
      return;
    }
    bind("real-fps", S.real.fps ? `camera ${S.real.fps.toFixed(0)} fps` : "– fps");
    const cam = state.src === "sim" ? S.sim : S.real;
    const fps = shownFps(state.src);
    const shown = fps == null ? "–" : fps.toFixed(fps < 10 ? 1 : 0);
    bind("cam-detail", state.src === "sim"
      ? `${config?.sim_camera ?? "overhead"} render · ${shown} fps shown · not twin-synced`
      : `D415 Top · ${shown} fps shown`);
    bind("cam-age", `age ${fmtAge(cam.age_s)}`);
    bind("cam-error", cam.error || (cam.ok ? "" : "The source sends no fresh frames."));
    const cams = Object.entries(S.cloud.cameras || {}).map(([n, c]) => `${n} ${c.ok ? c.points : "×"}`).join(" · ");
    bind("cloud-detail", `${cams} · age ${fmtAge(S.cloud.age_s)}`);
    bind("cloud-points", S.cloud.ok ? `${S.cloud.points.toLocaleString()} pts` : "down");
    bind("cloud-error", Object.entries(S.cloud.cameras || {}).filter(([, c]) => !c.ok).map(([n, c]) => `${n}: ${c.error}`).join(" · ") || "Waiting for depth.");
    const sim = S.scene.sim;
    bind("sim-state", !S.scene.ok ? "down" : sim?.busy ? "RUNNING" : sim?.stopped ? "STOPPED" : "IDLE");
    const f = focusOverlay();
    bind("focus-label", f ? (f.label || f.object_id) : "–");
    for (const kind of ["anomalies", "focus"]) {
      const o = status[kind];
      $(`[data-stub="${kind}"]`).classList.toggle("show", !o.connected && state[kind === "anomalies" ? "defects" : "focus"]);
      bind(`${kind}-reason`, o.reason || "");
    }
    bind("defect-count", String(visibleAnomalies().length));
  }

  // ------------------------------------------------------------------ controls
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-src],[data-opt],[data-scope],[data-fs],[data-color],[data-zoom]");
    if (!t) return;
    if (t.dataset.src) state.src = t.dataset.src;
    if (t.dataset.opt) state[t.dataset.opt] = !state[t.dataset.opt];
    if (t.dataset.scope) state.scope = t.dataset.scope;
    if (t.hasAttribute("data-zoom")) state.zoom = !state.zoom;
    if (t.dataset.color) { state.color = t.dataset.color; cloud?.recolor(); }
    if (t.hasAttribute("data-fs")) {
      const p = t.closest("[data-panel]").dataset.panel;
      state.fs = state.fs === p ? "" : p;
    }
    sync();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.target instanceof Element && e.target.matches("input,textarea")) || e.altKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if (k === "t") state.src = state.src === "sim" ? "real" : "sim";
    else if (k === "escape") state.fs = "";
    else if (k === "d") state.defects = !state.defects;
    else if (k === "f") state.focus = !state.focus;
    else if (k === "z") state.zoom = !state.zoom;
    else if (cloud && cloud.key(e)) return;
    else return;
    sync();
  });

  // ------------------------------------------------------------------ live camera frames
  const frames = { sim: { seq: 0, shown: [] }, real: { seq: 0, shown: [] } };
  const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
  async function frameLoop(name) {
    const img = $(`img.cam-${name}`), f = frames[name];
    let url = null;
    for (;;) {
      const visible = state.src === name;
      try {
        // visible: long-poll the next frame; hidden: take the newest frame once a second
        const r = await fetchT(`/frame/${name}.jpg?after=${visible ? f.seq : 0}`, 6000);
        if (r.status === 200) {
          const seq = Number(r.headers.get("X-Frame-Seq"));
          if (seq !== f.seq) {
            const next = URL.createObjectURL(await r.blob());
            img.src = next;
            f.seq = seq;
            // no await on decode(): it only resolves when the page paints
            const old = url;
            if (old) setTimeout(() => URL.revokeObjectURL(old), 2000);
            url = next;
            f.shown.push(performance.now());
            if (f.shown.length > 20) f.shown.shift();
          } else await r.arrayBuffer();
        } else if (r.status !== 204) await sleep(1000);
      } catch { await sleep(1000); }
      if (state.src !== name) await waitVisibleOr(1000, name);
    }
  }
  const wakers = [];
  function waitVisibleOr(ms, name) {
    return new Promise((ok) => { const t = setTimeout(ok, ms); wakers.push(() => { if (state.src === name) { clearTimeout(t); ok(); } }); });
  }
  function wake() { wakers.splice(0).forEach((w) => w()); }
  function shownFps(name) {
    const t = frames[name].shown;
    return t.length > 2 && performance.now() - t[t.length - 1] < 3000 ? ((t.length - 1) * 1000) / (t[t.length - 1] - t[0]) : null;
  }

  // ------------------------------------------------------------------ camera overlay
  const camPanel = $('[data-panel="cam"]');
  const overlay = $("canvas.cam-overlay", camPanel);
  function projectTop(p) {
    const c = config.cameras.top, R = c.rotation_cv, d = [p[0] - c.pos[0], p[1] - c.pos[1], p[2] - c.pos[2]];
    const q = [0, 1, 2].map((j) => R[0][j] * d[0] + R[1][j] * d[1] + R[2][j] * d[2]);
    return q[2] > 0 ? [c.fx * q[0] / q[2] + c.cx, c.fy * q[1] / q[2] + c.cy] : null;
  }
  function contentRect(stage) {
    const W = stage.clientWidth, H = stage.clientHeight, iw = config.image.width, ih = config.image.height;
    if (!state.zoom) {
      const s = Math.min(W / iw, H / ih);
      return { x: (W - iw * s) / 2, y: (H - ih * s) / 2, s };
    }
    // 2x: fill the stage, centred on the focus block, else on the plates (pivot)
    const s = Math.max(W / iw, H / ih) * 2.2;
    const f = focusOverlay();
    const [cx, cy] = f ? [(f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2] : projectTop(config.pivot) || [iw / 2, ih / 2];
    return { x: Math.min(0, Math.max(W - iw * s, W / 2 - cx * s)), y: Math.min(0, Math.max(H - ih * s, H / 2 - cy * s)), s };
  }
  function drawOverlay() {
    if (!config) return;
    const stage = overlay.parentElement, dpr = devicePixelRatio || 1;
    overlay.width = stage.clientWidth * dpr;
    overlay.height = stage.clientHeight * dpr;
    const g = overlay.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, overlay.width, overlay.height);
    const r = contentRect(stage);
    $$("img", stage).forEach((im) => Object.assign(im.style, {
      left: `${r.x}px`, top: `${r.y}px`, width: `${config.image.width * r.s}px`, height: `${config.image.height * r.s}px`,
    }));
    const P = (x, y) => [r.x + x * r.s, r.y + y * r.s];
    const st = {
      line: parseFloat(css(camPanel, "--box-line", "2")),
      kind: css(camPanel, "--box-style", "rect"),
      font: css(camPanel, "--label-font", "12px monospace"),
      fg: css(camPanel, "--label-fg", "#000"),
      pad: parseFloat(css(camPanel, "--box-pad", "4")),
    };
    const f = focusOverlay();
    if (state.focus && f) box(g, P, f.box, css(camPanel, "--focus", "#ff6a1a"), st, st.pad, `FOCUS  ${(f.label || f.object_id).toUpperCase()}`, "top");
    for (const a of visibleAnomalies()) {
      const label = [a.label, a.score != null ? a.score.toFixed(2) : null].filter(Boolean).join(" ");
      box(g, P, a.box, css(camPanel, "--defect", "#ff3b3b"), st, 2, label, "bottom");
    }
  }
  function box(g, P, b, color, st, pad, label, where) {
    let [x0, y0] = P(b[0], b[1]);
    let [x1, y1] = P(b[2], b[3]);
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
    g.strokeStyle = color;
    g.lineWidth = st.line;
    if (st.kind === "corners") {
      const c = Math.max(6, Math.min(x1 - x0, y1 - y0) * 0.28);
      g.beginPath();
      for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
        g.moveTo(x + dx * c, y); g.lineTo(x, y); g.lineTo(x, y + dy * c);
      }
      g.stroke();
    } else g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (!label) return;
    g.font = st.font;
    const px = parseFloat((st.font.match(/([\d.]+)px/) || [0, 12])[1]);
    const w = g.measureText(label).width + 10, h = px + 8, ly = where === "top" ? y0 - h - 2 : y1 + 2;
    g.fillStyle = color;
    g.fillRect(x0 - st.line / 2, ly, w, h);
    g.fillStyle = st.fg;
    g.textBaseline = "middle";
    g.fillText(label, x0 + 5 - st.line / 2, ly + h / 2 + 0.5);
  }
  new ResizeObserver(drawOverlay).observe(overlay.parentElement);

  // ------------------------------------------------------------------ point cloud (WebGL)
  function makeCloud() {
    const panel = $('[data-panel="depth"]');
    const canvas = $("canvas.cloud", panel), labels = $("canvas.cloud-labels", panel), gizmo = $("canvas.gizmo", panel);
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) { bind("cloud-error", "WebGL is not available in this browser."); return null; }
    const cams = config.cameras, top = cams.top, pivot = config.pivot;

    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, `attribute vec3 p;attribute vec3 c;uniform mat4 m;uniform float k;uniform float ortho;varying vec3 v;
      void main(){gl_Position=m*vec4(p,1.);gl_PointSize=clamp(ortho>.5?k:k/gl_Position.w,1.,8.);v=c;}`));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, "precision mediump float;varying vec3 v;void main(){gl_FragColor=vec4(v,1.);}"));
    gl.linkProgram(prog);
    const U = (n) => gl.getUniformLocation(prog, n), pa = gl.getAttribLocation(prog, "p"), ca = gl.getAttribLocation(prog, "c");
    const buf = (data, usage = gl.STATIC_DRAW) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage); return b; };

    // live cloud buffers
    const pBuf = gl.createBuffer(), cBuf = gl.createBuffer();
    let n = 0, rgb = new Float32Array(0), sensor = new Uint8Array(0);
    function setPoints(arrayBuffer) {
      // records: int16 x, y, z (mm); uint16 sensor << 15 | RGB555
      const xyz = new Int16Array(arrayBuffer), col = new Uint16Array(arrayBuffer), count = xyz.length / 4;
      const pos = new Float32Array(count * 3);
      rgb = new Float32Array(count * 3); sensor = new Uint8Array(count);
      for (let i = 0; i < count; i++) {
        const o = i * 4, c = col[o + 3];
        pos[i * 3] = xyz[o] / 1000; pos[i * 3 + 1] = xyz[o + 1] / 1000; pos[i * 3 + 2] = xyz[o + 2] / 1000;
        rgb[i * 3] = ((c >> 10) & 31) / 31; rgb[i * 3 + 1] = ((c >> 5) & 31) / 31; rgb[i * 3 + 2] = (c & 31) / 31;
        sensor[i] = c >> 15;
      }
      n = count;
      gl.bindBuffer(gl.ARRAY_BUFFER, pBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
      recolor();
    }
    function recolor() {
      const a = hex(css(panel, "--sensor-a", "#ff6a1a")), b = hex(css(panel, "--sensor-b", "#f1f1f0"));
      const out = state.color === "sensor" ? new Float32Array(n * 3) : rgb;
      if (state.color === "sensor") for (let i = 0; i < n; i++) out.set(sensor[i] ? b : a, i * 3);
      gl.bindBuffer(gl.ARRAY_BUFFER, cBuf); gl.bufferData(gl.ARRAY_BUFFER, out, gl.DYNAMIC_DRAW);
      dirty();
    }

    // reference lines: table grid and the D415 frustums (top camera last, hidden in camera view)
    let lineP, lineC, lineN, topStart = 0;
    (function buildLines() {
      const L = [], C = [], seg = (a, b, c) => { L.push(...a, ...b); C.push(...c, ...c); };
      const grid = hex(css(panel, "--cloud-grid", "#2a1a0f")), camc = hex(css(panel, "--cloud-cam", "#ff8a45"));
      const [[x0, x1], [y0, y1]] = config.workspace;
      for (let x = Math.ceil(x0 * 20) / 20; x <= x1 + 1e-9; x += 0.05) seg([x, y0, 0], [x, y1, 0], grid);
      for (let y = Math.ceil(y0 * 20) / 20; y <= y1 + 1e-9; y += 0.05) seg([x0, y, 0], [x1, y, 0], grid);
      for (const name of [...Object.keys(cams).filter((k) => k !== "top"), "top"]) {
        if (name === "top") topStart = L.length / 3;
        const c = cams[name], R = c.rotation_cv, s = 0.07, col = (j) => [R[0][j], R[1][j], R[2][j]];
        const [rx, ry, rz] = [col(0), col(1), col(2)];
        const corner = (a, b) => c.pos.map((p, i) => p + s * (rz[i] + a * 0.66 * rx[i] + b * 0.5 * ry[i]));
        const k = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
        k.forEach((q, i) => { seg(c.pos, q, camc); seg(q, k[(i + 1) % 4], camc); });
      }
      lineP = buf(new Float32Array(L)); lineC = buf(new Float32Array(C)); lineN = L.length / 3;
    })();

    // view: "cam" = exactly the real Top camera; "orbit" = Blender-style around a target
    const R = top.rotation_cv, col = (j) => [R[0][j], R[1][j], R[2][j]];
    const fovBase = 2 * Math.atan(config.image.height / 2 / top.fy), camAspect = config.image.width / config.image.height;
    let fovy = fovBase;
    const view = { mode: "cam", ortho: false, target: [...pivot], yaw: -Math.PI / 2, pitch: 0.6, dist: 0.85 };
    const sub = (a, b) => a.map((v, i) => v - b[i]), add = (a, b) => a.map((v, i) => v + b[i]), mul = (a, s) => a.map((v) => v * s);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => mul(a, 1 / Math.hypot(...a));
    function basis() {
      if (view.mode === "cam") return { eye: top.pos, right: col(0), up: mul(col(1), -1), back: mul(col(2), -1), dist: dot(sub(pivot, top.pos), col(2)) };
      const cp = Math.cos(view.pitch), back = [cp * Math.cos(view.yaw), cp * Math.sin(view.yaw), Math.sin(view.pitch)];
      const right = norm(cross([0, 0, 1], back));
      return { eye: add(view.target, mul(back, view.dist)), right, up: cross(back, right), back, dist: view.dist };
    }
    function leaveCam() {
      if (view.mode !== "cam") return;
      const fwd = col(2), t = dot(sub(pivot, top.pos), fwd);
      Object.assign(view, { target: add(top.pos, mul(fwd, t)), dist: t, yaw: Math.atan2(-fwd[1], -fwd[0]), pitch: Math.asin(-fwd[2]), mode: "orbit" });
    }
    function matrix(aspect) {
      const { eye, right, up, back, dist } = basis();
      fovy = aspect < camAspect ? 2 * Math.atan(Math.tan(fovBase / 2) * camAspect / aspect) : fovBase;
      const V = [right[0], up[0], back[0], 0, right[1], up[1], back[1], 0, right[2], up[2], back[2], 0, -dot(right, eye), -dot(up, eye), -dot(back, eye), 1];
      const near = 0.02, far = 5;
      let P;
      if (view.ortho) {
        const h = dist * Math.tan(fovy / 2), w = h * aspect;
        P = [1 / w, 0, 0, 0, 0, 1 / h, 0, 0, 0, 0, -2 / (far - near), 0, 0, 0, -(far + near) / (far - near), 1];
      } else {
        const f = 1 / Math.tan(fovy / 2);
        P = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0];
      }
      const M = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += P[k * 4 + r] * V[c * 4 + k]; M[c * 4 + r] = s; }
      return { M, dist };
    }

    let pending = false;
    function dirty() { if (!pending) { pending = true; requestAnimationFrame(draw); } }
    function resize() {
      const dpr = devicePixelRatio || 1;
      for (const c of [canvas, labels]) { c.width = canvas.clientWidth * dpr; c.height = canvas.clientHeight * dpr; }
      gizmo.width = gizmo.clientWidth * dpr; gizmo.height = gizmo.clientHeight * dpr;
      dirty();
    }
    function draw() {
      pending = false;
      const W = canvas.width, H = canvas.height;
      if (!W || !H) return;
      gl.viewport(0, 0, W, H);
      const bg = hex(css(panel, "--cloud-bg", "#0a0a0b"));
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(prog);
      const { M, dist } = matrix(W / H);
      gl.uniformMatrix4fv(U("m"), false, M);
      const pxPerM = H / (2 * Math.tan(fovy / 2)), size = parseFloat(css(panel, "--point-size", "0.0026"));
      gl.uniform1f(U("k"), view.ortho ? (size * pxPerM) / dist : size * pxPerM);
      gl.uniform1f(U("ortho"), view.ortho ? 1 : 0);
      gl.enableVertexAttribArray(pa); gl.enableVertexAttribArray(ca);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineP); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineC); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, view.mode === "cam" ? topStart : lineN);
      if (n) {
        gl.bindBuffer(gl.ARRAY_BUFFER, pBuf); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, cBuf); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, n);
      }
      drawLabels(M);
      drawGizmo();
      bind("proj", view.ortho ? "ORTHO" : "PERSP");
      bind("view", view.mode === "cam" ? "TOP CAMERA" : viewName());
      bind("dist", `${Math.round(dist * 1000)} mm`);
    }
    function viewName() {
      const d = (a, b) => Math.abs(a - b) < 0.02;
      if (view.pitch > 1.5) return "TOP";
      if (view.pitch < -1.5) return "BOTTOM";
      if (d(view.pitch, 0) && d(view.yaw, -Math.PI / 2)) return "FRONT";
      if (d(view.pitch, 0) && d(view.yaw, 0)) return "RIGHT";
      return "FREE";
    }
    function drawLabels(M) {
      const g = labels.getContext("2d"), dpr = devicePixelRatio || 1;
      g.clearRect(0, 0, labels.width, labels.height);
      g.font = `600 ${parseFloat(css(panel, "--cloud-label-size", "12")) * dpr}px ${css(panel, "--label-family", "monospace")}`;
      g.fillStyle = css(panel, "--cloud-text", "#ffb27f");
      const put = (p, t) => {
        const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12], y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13], w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
        if (w > 0) g.fillText(t, (x / w * 0.5 + 0.5) * labels.width + 6 * dpr, (0.5 - y / w * 0.5) * labels.height - 6 * dpr);
      };
      if (view.mode !== "cam") put(top.pos, "D415 TOP");
      if (cams.plate) put(cams.plate.pos, "D415 PLATE");
    }
    function drawGizmo() {
      const g = gizmo.getContext("2d"), W = gizmo.width, H = gizmo.height, dpr = devicePixelRatio || 1;
      g.clearRect(0, 0, W, H);
      const { right, up, back } = basis(), c = [W / 2, H / 2], r = Math.min(W, H) * 0.34, items = [];
      for (const [name, a, v, fb] of [["X", [1, 0, 0], "--gx", "#ff5a5f"], ["Y", [0, 1, 0], "--gy", "#8fd14f"], ["Z", [0, 0, 1], "--gz", "#4d9cff"]])
        for (const s of [1, -1]) {
          const d = mul(a, s);
          items.push({ name, s, color: css(panel, v, fb), x: c[0] + dot(d, right) * r, y: c[1] - dot(d, up) * r, z: dot(d, back) });
        }
      items.sort((a, b) => a.z - b.z);
      gizmo._items = items;
      for (const it of items) {
        if (it.s > 0) { g.strokeStyle = it.color; g.lineWidth = 2 * dpr; g.beginPath(); g.moveTo(...c); g.lineTo(it.x, it.y); g.stroke(); }
        g.beginPath(); g.arc(it.x, it.y, (it.s > 0 ? 9 : 6.5) * dpr, 0, 7);
        g.fillStyle = it.s > 0 ? it.color : "transparent"; g.fill();
        g.strokeStyle = it.color; g.lineWidth = 1.5 * dpr; g.stroke();
        if (it.s > 0) { g.fillStyle = "#000"; g.font = `bold ${10 * dpr}px sans-serif`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(it.name, it.x, it.y + 0.5); }
      }
    }

    // Blender-like interaction (full screen only)
    function preset(name) {
      if (name === "cam") { view.mode = "cam"; view.ortho = false; return dirty(); }
      leaveCam();
      view.target = [...pivot];
      const P = { top: [-Math.PI / 2, Math.PI / 2 - 1e-4], bottom: [-Math.PI / 2, -Math.PI / 2 + 1e-4], front: [-Math.PI / 2, 0], back: [Math.PI / 2, 0], right: [0, 0], left: [Math.PI, 0] }[name];
      if (!P) return;
      [view.yaw, view.pitch] = P;
      view.dist = 0.85;
      dirty();
    }
    const orbit = (dx, dy) => { leaveCam(); view.yaw -= dx * 0.008; view.pitch = Math.max(-1.5707, Math.min(1.5707, view.pitch + dy * 0.008)); dirty(); };
    const pan = (dx, dy) => {
      leaveCam();
      const { right, up } = basis(), k = (2 * view.dist * Math.tan(fovy / 2)) / canvas.clientHeight;
      view.target = add(view.target, add(mul(right, -dx * k), mul(up, dy * k)));
      dirty();
    };
    const zoom = (f) => { leaveCam(); view.dist = Math.max(0.08, Math.min(3, view.dist * f)); dirty(); };
    let drag = null;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
      if (!panel.classList.contains("fs")) return;
      drag = { how: e.shiftKey || e.button === 2 ? "pan" : e.ctrlKey ? "zoom" : "orbit", x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    $$("[data-drag]", panel).forEach((h) => h.addEventListener("pointerdown", (e) => {
      drag = { how: h.dataset.drag, x: e.clientX, y: e.clientY };
      h.setPointerCapture(e.pointerId);
      e.preventDefault();
    }));
    window.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.how === "orbit") orbit(dx, dy); else if (drag.how === "pan") pan(dx, dy); else zoom(Math.exp(dy * 0.01));
    });
    window.addEventListener("pointerup", () => (drag = null));
    canvas.addEventListener("wheel", (e) => { if (!panel.classList.contains("fs")) return; e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0012)); }, { passive: false });
    gizmo.addEventListener("click", (e) => {
      const r = gizmo.getBoundingClientRect(), dpr = devicePixelRatio || 1, x = (e.clientX - r.left) * dpr, y = (e.clientY - r.top) * dpr;
      const hit = (gizmo._items || []).filter((i) => Math.hypot(i.x - x, i.y - y) < 12 * dpr).sort((a, b) => b.z - a.z)[0];
      if (hit) preset({ X1: "right", "X-1": "left", Y1: "back", "Y-1": "front", Z1: "top", "Z-1": "bottom" }[hit.name + hit.s]);
    });
    document.addEventListener("click", (e) => {
      const v = e.target.closest("[data-view]");
      if (v) preset(v.dataset.view);
      if (e.target.closest("[data-proj]")) { leaveCam(); view.ortho = !view.ortho; dirty(); }
    });
    function key(e) {
      if (state.fs !== "depth") return false;
      const map = { 1: e.ctrlKey ? "back" : "front", 3: e.ctrlKey ? "left" : "right", 7: e.ctrlKey ? "bottom" : "top", 0: "cam" };
      if (map[e.key]) { e.preventDefault(); preset(map[e.key]); return true; }
      if (e.key === "5") { leaveCam(); view.ortho = !view.ortho; dirty(); return true; }
      if (e.key === "Home" || e.key === ".") { leaveCam(); view.target = [...pivot]; view.dist = 0.85; dirty(); return true; }
      return false;
    }
    new ResizeObserver(resize).observe(canvas);
    if (hash.view) preset(hash.view);
    if (hash.orbit) { leaveCam(); const [y, p, d] = hash.orbit.split(",").map(Number); Object.assign(view, { yaw: y, pitch: p, dist: d, target: [...pivot] }); }
    return { resize, recolor, key, dirty, setPoints, count: () => n, view };
  }

  // ------------------------------------------------------------------ live loops
  async function loadConfig() {
    for (;;) {
      try {
        const r = await fetchT("/api/config", 5000);
        if (r.ok) return await r.json();
        bind("health", "Waiting for sim scene");
      } catch { bind("health", "Server unreachable"); }
      await new Promise((ok) => setTimeout(ok, 1500));
    }
  }
  async function statusLoop() {
    for (;;) {
      try {
        const r = await fetchT("/api/status", 3000);
        if (r.ok) { status = await r.json(); statusAt = performance.now(); }
      } catch { /* statusFresh() turns false after STATUS_STALE_MS */ }
      renderStatus();
      drawOverlay();
      await new Promise((ok) => setTimeout(ok, 1000));
    }
  }
  const cloudState = { seq: 0, updates: 0 };
  async function cloudLoop() {
    for (;;) {
      try {
        const r = await fetchT(`/api/cloud?after=${cloudState.seq}`, 8000);
        if (r.status === 200) {
          cloudState.seq = Number(r.headers.get("X-Cloud-Seq"));
          cloud.setPoints(await r.arrayBuffer());
          cloudState.updates++;
        } else if (r.status !== 204) await new Promise((ok) => setTimeout(ok, 1000));
      } catch { await new Promise((ok) => setTimeout(ok, 1000)); }
    }
  }

  // ------------------------------------------------------------------ icons + clock
  const ICON = {
    fullscreen: '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
    orbit: '<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v4.5h-4.5"/>',
    pan: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
    zoom: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5M10.5 8v5"/>',
  };
  $$("[data-icon]").forEach((e) => e.insertAdjacentHTML("afterbegin",
    `<svg viewBox="0 0 24 24" width="${e.dataset.iconSize || 16}" height="${e.dataset.iconSize || 16}" fill="none" stroke="currentColor" stroke-width="${e.dataset.iconStroke || 1.6}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[e.dataset.icon]}</svg>`));
  const tick = () => bind("clock", new Date().toTimeString().slice(0, 8));
  tick();
  setInterval(tick, 1000);

  sync();
  loadConfig().then((c) => {
    config = c;
    frameLoop("sim");
    frameLoop("real");
    cloud = makeCloud();
    window.dashboard = { state, get status() { return status; }, get config() { return config; }, cloudState, cloud, sync, frames, shownFps };
    sync();
    statusLoop();
    if (cloud) cloudLoop();
  });
})();
