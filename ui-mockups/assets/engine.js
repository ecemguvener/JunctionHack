/* Shared mockup engine. Each design supplies the markup and CSS; this file supplies behaviour.
 *
 * Markup contract (attributes, not classes, so each design can style freely):
 *   [data-panel="cam"]   .stage > img.cam-sim + img.cam-real + canvas.cam-overlay
 *   [data-panel="depth"] .stage > canvas.cloud + canvas.cloud-labels ; .nav > canvas.gizmo
 *   [data-src=sim|real]  source buttons (any number)       key T toggles
 *   [data-fs]            full-screen button in a panel      key Esc leaves
 *   [data-opt=defects|focus]  option toggles               keys D / F
 *   [data-scope=all|focus]    defect scope
 *   [data-view=cam|top|front|right] [data-proj] [data-drag=orbit|pan|zoom] [data-color=rgb|sensor]
 *   [data-bind=...]      text outputs
 * Body classes mirror the state: src-sim/src-real, opt-defects, opt-focus, scope-all/scope-focus,
 * fs-cam/fs-depth, proj-ortho. Overlay look comes from CSS variables on the cam panel.
 */
(() => {
  const T = window.TWIN;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (el, name, fallback) =>
    getComputedStyle(el).getPropertyValue(name).trim() || fallback;

  const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  const state = {
    src: hash.src || "sim",
    defects: hash.defects === "1",
    focus: hash.focus !== "0",
    scope: hash.scope || "all",
    fs: hash.fs || "",
    color: hash.color || "rgb",
    zoom: hash.zoom === "2",
  };

  // ------------------------------------------------------------------ state → DOM
  function sync() {
    const b = document.body.classList;
    b.toggle("src-sim", state.src === "sim");
    b.toggle("src-real", state.src === "real");
    b.toggle("opt-defects", state.defects);
    b.toggle("opt-focus", state.focus);
    b.toggle("scope-all", state.scope === "all");
    b.toggle("scope-focus", state.scope === "focus");
    b.toggle("fs-cam", state.fs === "cam");
    b.toggle("fs-depth", state.fs === "depth");
    b.toggle("zoomed", state.zoom);
    $$("[data-src]").forEach((e) => setOn(e, e.dataset.src === state.src));
    $$("[data-opt]").forEach((e) => setOn(e, state[e.dataset.opt]));
    $$("[data-scope]").forEach((e) => setOn(e, e.dataset.scope === state.scope));
    $$("[data-color]").forEach((e) => setOn(e, e.dataset.color === state.color));
    $$("[data-zoom]").forEach((e) => setOn(e, state.zoom));
    bind("zoom", state.zoom ? "2×" : "1×");
    $$("[data-panel]").forEach((p) => p.classList.toggle("fs", p.dataset.panel === state.fs));
    $$("[data-fs]").forEach((e) => setOn(e, e.closest("[data-panel]")?.dataset.panel === state.fs));
    bind("src", state.src === "sim" ? "SIM" : "REAL");
    bind("src-long", state.src === "sim" ? "Digital twin" : "Real camera");
    const focus = T.bricks.find((b) => b.id === T.focus);
    bind("focus-label", focus ? focus.label : "none");
    bind("focus-id", T.focus);
    bind("defect-count", String(visibleAnomalies().length));
    bind("threshold", T.threshold.toFixed(2));
    drawOverlay();
    cloud?.resize();
  }
  function setOn(el, on) {
    el.classList.toggle("on", !!on);
    el.setAttribute("aria-pressed", on ? "true" : "false");
    if (el.type === "checkbox") el.checked = !!on;
  }
  function bind(name, text) {
    $$(`[data-bind="${name}"]`).forEach((e) => (e.textContent = text));
  }
  const visibleAnomalies = () =>
    state.defects ? T.anomalies.filter((a) => state.scope === "all" || a.brick === T.focus) : [];

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
    if (e.target.matches("input,textarea")) return;
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

  // ------------------------------------------------------------------ camera overlay
  const camPanel = $('[data-panel="cam"]');
  const overlay = camPanel && $("canvas.cam-overlay", camPanel);
  function contentRect(stage) {
    const W = stage.clientWidth, H = stage.clientHeight, iw = T.image.width, ih = T.image.height;
    if (!state.zoom) {
      const s = Math.min(W / iw, H / ih);
      return { x: (W - iw * s) / 2, y: (H - ih * s) / 2, s };
    }
    // 2x: fill the stage and centre on the focus block, clamped to the image
    const s = Math.max(W / iw, H / ih) * 2.2 / (W / H > iw / ih ? 1 : 1);
    const f = T.bricks.find((b) => b.id === T.focus).box;
    const cx = (f[0] + f[2]) / 2, cy = (f[1] + f[3]) / 2;
    const x = Math.min(0, Math.max(W - iw * s, W / 2 - cx * s)), y = Math.min(0, Math.max(H - ih * s, H / 2 - cy * s));
    return { x, y, s };
  }
  function drawOverlay() {
    if (!overlay) return;
    const stage = overlay.parentElement, dpr = devicePixelRatio || 1;
    overlay.width = stage.clientWidth * dpr;
    overlay.height = stage.clientHeight * dpr;
    const g = overlay.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, overlay.width, overlay.height);
    const r = contentRect(stage);
    $$("img", stage).forEach((im) => Object.assign(im.style, {
      left: `${r.x}px`, top: `${r.y}px`, width: `${T.image.width * r.s}px`, height: `${T.image.height * r.s}px`,
    }));
    const P = (x, y) => [r.x + x * r.s, r.y + y * r.s];
    const style = {
      line: parseFloat(css(camPanel, "--box-line", "1.5")),
      kind: css(camPanel, "--box-style", "rect"),
      font: css(camPanel, "--label-font", "11px monospace"),
      labelFg: css(camPanel, "--label-fg", "#000"),
      pad: parseFloat(css(camPanel, "--box-pad", "4")),
    };
    if (state.focus) {
      const b = T.bricks.find((b) => b.id === T.focus);
      const color = css(camPanel, "--focus", "#ff6a00");
      box(g, P, b.box, color, style, style.pad, `FOCUS  ${b.label.toUpperCase()}`, "top");
    }
    visibleAnomalies().forEach((a) => {
      const color = css(camPanel, "--defect", "#ff2d2d");
      box(g, P, a.box, color, style, 2, `${a.score.toFixed(2)}`, "bottom");
    });
  }
  function box(g, P, b, color, st, pad, label, where) {
    let [x0, y0] = P(b[0], b[1]);
    let [x1, y1] = P(b[2], b[3]);
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
    g.strokeStyle = color;
    g.lineWidth = st.line;
    g.setLineDash(st.kind === "dashed" ? [4, 3] : []);
    if (st.kind === "corners") {
      const c = Math.max(5, Math.min(x1 - x0, y1 - y0) * 0.28);
      g.beginPath();
      for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
        g.moveTo(x + dx * c, y); g.lineTo(x, y); g.lineTo(x, y + dy * c);
      }
      g.stroke();
    } else if (st.kind === "round") {
      g.beginPath(); g.roundRect(x0, y0, x1 - x0, y1 - y0, 6); g.stroke();
    } else {
      g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
    if (!label) return;
    g.setLineDash([]);
    g.font = st.font;
    const px = parseFloat((st.font.match(/([\d.]+)px/) || [0, 11])[1]);
    const w = g.measureText(label).width + 10, h = px + 8;
    const ly = where === "top" ? y0 - h - 2 : y1 + 2;
    g.fillStyle = color;
    g.fillRect(x0 - st.line / 2, ly, w, h);
    g.fillStyle = st.labelFg;
    g.textBaseline = "middle";
    g.fillText(label, x0 + 5 - st.line / 2, ly + h / 2 + 0.5);
  }
  if (overlay) new ResizeObserver(drawOverlay).observe(overlay.parentElement);

  // ------------------------------------------------------------------ point cloud (WebGL)
  const cloud = (() => {
    const panel = $('[data-panel="depth"]');
    const canvas = panel && $("canvas.cloud", panel);
    if (!canvas) return null;
    const gl = canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: true });
    if (!gl) return null;
    const labels = $("canvas.cloud-labels", panel);
    const gizmo = $("canvas.gizmo", panel);

    // decode int16 x5 records
    const raw = Uint8Array.from(atob(T.cloud.b64), (c) => c.charCodeAt(0));
    const rec = new Int16Array(raw.buffer);
    const n = T.cloud.n, pos = new Float32Array(n * 3), rgb = new Float32Array(n * 3), sensor = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const o = i * 5;
      pos[i * 3] = rec[o] / 1000; pos[i * 3 + 1] = rec[o + 1] / 1000; pos[i * 3 + 2] = rec[o + 2] / 1000;
      const c = rec[o + 3];
      rgb[i * 3] = ((c >> 10) & 31) / 31; rgb[i * 3 + 1] = ((c >> 5) & 31) / 31; rgb[i * 3 + 2] = (c & 31) / 31;
      sensor[i] = rec[o + 4];
    }
    const hex = (h) => { const v = parseInt(h.replace("#", ""), 16); return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255]; };

    const prog = (() => {
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, `attribute vec3 p;attribute vec3 c;uniform mat4 m;uniform float k;uniform float ortho;varying vec3 v;
        void main(){gl_Position=m*vec4(p,1.);gl_PointSize=clamp(ortho>.5?k:k/gl_Position.w,1.,7.);v=c;}`));
      gl.attachShader(p, sh(gl.FRAGMENT_SHADER, `precision mediump float;varying vec3 v;void main(){gl_FragColor=vec4(v,1.);}`));
      gl.linkProgram(p);
      return p;
    })();
    const buf = (data) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); return b; };
    const pBuf = buf(pos);
    let cBuf = null;
    function recolor() {
      const a = hex(css(panel, "--sensor-a", "#ff7a1a")), b = hex(css(panel, "--sensor-b", "#2fb6ff"));
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const s = state.color === "sensor" ? (sensor[i] ? b : a) : [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]];
        out.set(s, i * 3);
      }
      if (cBuf) gl.deleteBuffer(cBuf);
      cBuf = buf(out);
      dirty();
    }

    // reference lines: table grid, plate rings, camera frustums
    let lineP, lineC, lineN, topStart = 0;
    function buildLines() {
      const L = [], C = [];
      const grid = hex(css(panel, "--cloud-grid", "#333333")), ring = hex(css(panel, "--cloud-ring", "#777777"));
      const camc = hex(css(panel, "--cloud-cam", "#aaaaaa"));
      const seg = (a, b, c) => { L.push(...a, ...b); C.push(...c, ...c); };
      for (let x = -0.1; x <= 0.6001; x += 0.05) seg([x, -0.5, 0], [x, 0.4, 0], grid);
      for (let y = -0.5; y <= 0.4001; y += 0.05) seg([-0.1, y, 0], [0.6, y, 0], grid);
      for (const name in T.plates) {
        const [px, py] = T.plates[name];
        for (const [r, z] of [[0.095, 0.022], [0.07, 0.004]])
          for (let i = 0; i < 64; i++) {
            const a0 = (i / 64) * 2 * Math.PI, a1 = ((i + 1) / 64) * 2 * Math.PI;
            seg([px + r * Math.cos(a0), py + r * Math.sin(a0), z], [px + r * Math.cos(a1), py + r * Math.sin(a1), z], ring);
          }
      }
      for (const name of [...Object.keys(T.cameras).filter((k) => k !== "top"), "top"]) {
        if (name === "top") topStart = L.length / 3;
        const c = T.cameras[name], R = c.rotation_cv, s = 0.07;
        const col = (j) => [R[0][j], R[1][j], R[2][j]];
        const [rx, ry, rz] = [col(0), col(1), col(2)];
        const corner = (a, b) => c.pos.map((p, i) => p + s * (rz[i] + a * 0.66 * rx[i] + b * 0.5 * ry[i]));
        const k = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
        k.forEach((q, i) => { seg(c.pos, q, camc); seg(q, k[(i + 1) % 4], camc); });
      }
      lineP = buf(new Float32Array(L)); lineC = buf(new Float32Array(C)); lineN = L.length / 3;
    }

    // ---- view state
    const top = T.cameras.top;
    const R = top.rotation_cv, col = (j) => [R[0][j], R[1][j], R[2][j]];
    const fovBase = 2 * Math.atan(T.image.height / 2 / top.fy);
    let lastFovy = fovBase;
    const view = { mode: "cam", ortho: false, target: [...T.pivot], yaw: -Math.PI / 2, pitch: 0.6, dist: 0.8 };
    const sub = (a, b) => a.map((v, i) => v - b[i]), add = (a, b) => a.map((v, i) => v + b[i]);
    const mul = (a, s) => a.map((v) => v * s), dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => mul(a, 1 / Math.hypot(...a));

    function basis() {
      if (view.mode === "cam") {
        return { eye: top.pos, right: col(0), up: mul(col(1), -1), back: mul(col(2), -1), dist: dot(sub(T.pivot, top.pos), col(2)) };
      }
      const cp = Math.cos(view.pitch), back = [cp * Math.cos(view.yaw), cp * Math.sin(view.yaw), Math.sin(view.pitch)];
      const right = norm(cross([0, 0, 1], back)), up = cross(back, right);
      return { eye: add(view.target, mul(back, view.dist)), right, up, back, dist: view.dist };
    }
    function leaveCam() {
      if (view.mode !== "cam") return;
      const fwd = col(2), t = dot(sub(T.pivot, top.pos), fwd);
      view.target = add(top.pos, mul(fwd, t));
      view.dist = t;
      view.yaw = Math.atan2(-fwd[1], -fwd[0]);
      view.pitch = Math.asin(-fwd[2]);
      view.mode = "orbit";
    }
    function matrix(aspect) {
      const { eye, right, up, back, dist } = basis();
      const camAspect = T.image.width / T.image.height;
      const fovy = aspect < camAspect ? 2 * Math.atan(Math.tan(fovBase / 2) * camAspect / aspect) : fovBase;
      lastFovy = fovy;
      const V = [right[0], up[0], back[0], 0, right[1], up[1], back[1], 0, right[2], up[2], back[2], 0,
        -dot(right, eye), -dot(up, eye), -dot(back, eye), 1];
      let P;
      const near = 0.02, far = 5;
      if (view.ortho) {
        const h = dist * Math.tan(fovy / 2), w = h * aspect;
        P = [1 / w, 0, 0, 0, 0, 1 / h, 0, 0, 0, 0, -2 / (far - near), 0, 0, 0, -(far + near) / (far - near), 1];
      } else {
        const f = 1 / Math.tan(fovy / 2);
        P = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0];
      }
      const M = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0; for (let k = 0; k < 4; k++) s += P[k * 4 + r] * V[c * 4 + k]; M[c * 4 + r] = s;
      }
      return { M, dist };
    }

    let pending = false;
    function dirty() { if (!pending) { pending = true; requestAnimationFrame(draw); } }
    function resize() {
      const dpr = devicePixelRatio || 1;
      for (const c of [canvas, labels]) if (c) { c.width = canvas.clientWidth * dpr; c.height = canvas.clientHeight * dpr; }
      if (gizmo) { gizmo.width = gizmo.clientWidth * dpr; gizmo.height = gizmo.clientHeight * dpr; }
      dirty();
    }
    function draw() {
      pending = false;
      const W = canvas.width, H = canvas.height;
      if (!W || !H) return;
      gl.viewport(0, 0, W, H);
      const bg = hex(css(panel, "--cloud-bg", "#111111"));
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(prog);
      const { M, dist } = matrix(W / H);
      gl.uniformMatrix4fv(gl.getUniformLocation(prog, "m"), false, M);
      const pxPerM = H / (2 * Math.tan(lastFovy / 2));
      const size = parseFloat(css(panel, "--point-size", "0.0024"));
      gl.uniform1f(gl.getUniformLocation(prog, "k"), view.ortho ? (size * pxPerM) / dist : size * pxPerM);
      gl.uniform1f(gl.getUniformLocation(prog, "ortho"), view.ortho ? 1 : 0);
      const pa = gl.getAttribLocation(prog, "p"), ca = gl.getAttribLocation(prog, "c");
      gl.enableVertexAttribArray(pa); gl.enableVertexAttribArray(ca);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineP); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineC); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, view.mode === "cam" ? topStart : lineN);
      gl.bindBuffer(gl.ARRAY_BUFFER, pBuf); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, cBuf); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.POINTS, 0, n);
      drawLabels(M);
      drawGizmo();
      bind("proj", view.ortho ? "ORTHO" : "PERSP");
      bind("view", view.mode === "cam" ? "TOP CAMERA" : viewName());
      bind("dist", `${Math.round(dist * 1000)} mm`);
    }
    function viewName() {
      const d = (a, b) => Math.abs(a - b) < 0.02;
      if (view.pitch > 1.5) return "TOP";
      if (d(view.pitch, 0) && d(view.yaw, -Math.PI / 2)) return "FRONT";
      if (d(view.pitch, 0) && d(view.yaw, 0)) return "RIGHT";
      return "FREE";
    }
    function project(M, p) {
      const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12], y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13];
      const w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
      return w <= 0 ? null : [(x / w * 0.5 + 0.5) * labels.width, (0.5 - y / w * 0.5) * labels.height];
    }
    function drawLabels(M) {
      if (!labels) return;
      const g = labels.getContext("2d"), dpr = devicePixelRatio || 1;
      g.clearRect(0, 0, labels.width, labels.height);
      g.font = `600 ${parseFloat(css(panel, "--cloud-label-size", "10")) * dpr}px ${css(panel, "--label-family", "monospace")}`;
      g.fillStyle = css(panel, "--cloud-text", "#999");
      const put = (p, t) => { const s = project(M, p); if (s) g.fillText(t, s[0] + 6 * dpr, s[1] - 6 * dpr); };
      for (const name in T.plates) put([T.plates[name][0] + 0.07, T.plates[name][1] - 0.07, 0.02], `PLATE · ${name.toUpperCase()}`);
      if (view.mode !== "cam") put(T.cameras.top.pos, "D415 TOP");
      put(T.cameras.plate.pos, "D415 PLATE");
    }
    function drawGizmo() {
      if (!gizmo) return;
      const g = gizmo.getContext("2d"), W = gizmo.width, H = gizmo.height, dpr = devicePixelRatio || 1;
      g.clearRect(0, 0, W, H);
      const { right, up, back } = basis(), c = [W / 2, H / 2], r = Math.min(W, H) * 0.34;
      const axes = [["X", [1, 0, 0], "--gx", "#e5484d"], ["Y", [0, 1, 0], "--gy", "#7bbf3a"], ["Z", [0, 0, 1], "--gz", "#3e8ce0"]];
      const items = [];
      for (const [name, a, v, fb] of axes) for (const s of [1, -1]) {
        const d = mul(a, s);
        items.push({ name, s, d, color: css(panel, v, fb), x: c[0] + dot(d, right) * r, y: c[1] - dot(d, up) * r, z: dot(d, back) });
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

    // ---- interaction (Blender-like)
    function preset(name) {
      if (name === "cam") { view.mode = "cam"; view.ortho = false; return dirty(); }
      leaveCam();
      view.target = [...T.pivot];
      if (name === "top") { view.yaw = -Math.PI / 2; view.pitch = Math.PI / 2 - 1e-4; }
      if (name === "front") { view.yaw = -Math.PI / 2; view.pitch = 0; }
      if (name === "back") { view.yaw = Math.PI / 2; view.pitch = 0; }
      if (name === "right") { view.yaw = 0; view.pitch = 0; }
      if (name === "left") { view.yaw = Math.PI; view.pitch = 0; }
      if (name === "bottom") { view.yaw = -Math.PI / 2; view.pitch = -Math.PI / 2 + 1e-4; }
      view.dist = 0.85;
      dirty();
    }
    function orbit(dx, dy) { leaveCam(); view.yaw -= dx * 0.008; view.pitch = Math.max(-1.5707, Math.min(1.5707, view.pitch + dy * 0.008)); dirty(); }
    function pan(dx, dy) {
      leaveCam();
      const { right, up } = basis(), k = (2 * view.dist * Math.tan(lastFovy / 2)) / canvas.clientHeight;
      view.target = add(view.target, add(mul(right, -dx * k), mul(up, dy * k)));
      dirty();
    }
    function zoom(f) { leaveCam(); view.dist = Math.max(0.08, Math.min(3, view.dist * f)); dirty(); }

    let drag = null;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
      if (!panel.classList.contains("fs")) return;
      const how = e.shiftKey ? "pan" : e.ctrlKey ? "zoom" : e.button === 2 ? "pan" : "orbit";
      drag = { how, x: e.clientX, y: e.clientY };
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
      if (drag.how === "orbit") orbit(dx, dy);
      if (drag.how === "pan") pan(dx, dy);
      if (drag.how === "zoom") zoom(Math.exp(dy * 0.01));
    });
    window.addEventListener("pointerup", () => (drag = null));
    canvas.addEventListener("wheel", (e) => {
      if (!panel.classList.contains("fs")) return;
      e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0012));
    }, { passive: false });
    gizmo?.addEventListener("click", (e) => {
      const r = gizmo.getBoundingClientRect(), dpr = devicePixelRatio || 1;
      const x = (e.clientX - r.left) * dpr, y = (e.clientY - r.top) * dpr;
      const hit = (gizmo._items || []).filter((i) => Math.hypot(i.x - x, i.y - y) < 11 * dpr).sort((a, b) => b.z - a.z)[0];
      if (!hit) return;
      const map = { "X1": "right", "X-1": "left", "Y1": "back", "Y-1": "front", "Z1": "top", "Z-1": "bottom" };
      preset(map[hit.name + hit.s]);
    });
    document.addEventListener("click", (e) => {
      const v = e.target.closest("[data-view]"); if (v) preset(v.dataset.view);
      if (e.target.closest("[data-proj]")) { leaveCam(); view.ortho = !view.ortho; dirty(); }
    });
    function key(e) {
      if (state.fs !== "depth") return false;
      const map = { "1": e.ctrlKey ? "back" : "front", "3": e.ctrlKey ? "left" : "right", "7": e.ctrlKey ? "bottom" : "top", "0": "cam" };
      if (map[e.key]) { e.preventDefault(); preset(map[e.key]); return true; }
      if (e.key === "5") { leaveCam(); view.ortho = !view.ortho; dirty(); return true; }
      if (e.key === "Home" || e.key === ".") { leaveCam(); view.target = [...T.pivot]; view.dist = 0.85; dirty(); return true; }
      return false;
    }

    buildLines();
    recolor();
    new ResizeObserver(resize).observe(canvas);
    if (hash.view) preset(hash.view);
    if (hash.orbit) { leaveCam(); const [y, p, d] = hash.orbit.split(",").map(Number); view.yaw = y; view.pitch = p; view.dist = d; view.target = [...T.pivot]; }
    return { resize, recolor, key, dirty };
  })();

  // ------------------------------------------------------------------ icons
  const ICON = {
    fullscreen: '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
    orbit: '<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v4.5h-4.5"/>',
    pan: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
    zoom: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5M10.5 8v5"/>',
    camera: '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
  };
  $$("[data-icon]").forEach((e) => {
    e.insertAdjacentHTML("afterbegin", `<svg viewBox="0 0 24 24" width="${e.dataset.iconSize || 16}" height="${e.dataset.iconSize || 16}" fill="none" stroke="currentColor" stroke-width="${e.dataset.iconStroke || 1.6}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[e.dataset.icon]}</svg>`);
  });

  // ------------------------------------------------------------------ live clock
  const tick = () => {
    const d = new Date();
    bind("clock", d.toTimeString().slice(0, 8));
    bind("date", d.toISOString().slice(0, 10));
  };
  tick();
  setInterval(tick, 1000);
  sync();
})();
