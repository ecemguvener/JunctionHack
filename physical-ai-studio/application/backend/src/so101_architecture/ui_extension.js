/**
 * Physical AI Studio — SO-101 Model-Based Pick-and-Place Architecture Workbench Extension
 *
 * Seamlessly integrates into Intel's Physical AI Studio UI (both via React route and
 * runtime SPA tab injection) to provide:
 *   1. All-Camera Feeds (Top D415, Side/Plate D415, Wrist Sonix UVC) via Studio WS or HTTP
 *   2. Leader & Follower SO-101 Arm Calibration Sync, Live Observation WS & Waypoint Teaching
 *   3. Exact Forward Kinematics (4x4 SE(3) in Base & World frames)
 *   4. 6x5 Geometric Jacobians (rad, STS3215 ticks, LeRobot degrees) + Manipulability w5(q)
 *   5. Published CAD Dynamics (5x5 Mass Matrix M(q), Gravity Torque g(q), COM, Stall Load %)
 *   6. Pure-NumPy 5-DOF Grasp/Placement Inverse Kinematics Solver
 *   7. 6-Stage Deterministic Pick -> Plate-Camera Spin -> Sort Trajectory Synthesis & Export
 */
(function () {
  "use strict";

  if (window.__SO101_ARCH_EXTENSION_LOADED__) return;
  window.__SO101_ARCH_EXTENSION_LOADED__ = true;

  const MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"];
  const WAYPOINT_ORDER = [
    "observation_home",
    "source_pregrasp",
    "source_pick",
    "plate_camera_spin",
    "good_bowl_drop",
    "defective_bowl_drop",
  ];

  const state = {
    open: false,
    snap: null,
    activeArm: "follower",
    unitMode: "q_enc_rad", // "q_enc_rad" | "ticks" | "lerobot_deg" | "normalized"
    jacMode: "J_world_rad", // "J_world_rad" | "J_base_rad" | "J_world_tick" | "J_world_lerobot_deg"
    fkFrame: "world", // "world" | "base"
    camStreamActive: true,
    camStreamTimer: null,
    leaderWs: null,
    leaderWsConnected: false,
    playingTraj: false,
    trajFrameIdx: 0,
    trajTimer: null,
    orbitAz: -0.65,
    orbitEl: 0.42,
    orbitDist: 0.85,
    dragging3d: false,
    lastMouse: { x: 0, y: 0 },
    lastIkResult: null,
    lastAlignResult: null,
  };

  function injectStyles() {
    if (document.getElementById("so101-arch-styles")) return;
    const style = document.createElement("style");
    style.id = "so101-arch-styles";
    style.textContent = `
      .so101-nav-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #0054ae 0%, #0071c5 100%);
        color: #ffffff !important;
        border: 1px solid #38bdf8;
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0, 113, 197, 0.35);
        transition: all 0.15s ease;
        z-index: 9999;
        white-space: nowrap;
      }
      .so101-nav-btn.floating-dock {
        position: fixed;
        bottom: 6px;
        right: 16px;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 11.5px;
      }
      .so101-nav-btn:hover {
        background: linear-gradient(135deg, #0066cc 0%, #0ea5e9 100%);
        transform: translateY(-1px);
      }
      .so101-nav-btn.active {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        border-color: #fdba74;
      }
      .so101-badge {
        background: rgba(255,255,255,0.18);
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 10px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
      }
      #so101-workbench-overlay {
        position: fixed;
        top: 64px;
        left: 0;
        right: 0;
        bottom: 32px;
        background: #111318;
        color: #e2e8f0;
        z-index: 9500;
        overflow-y: auto;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        padding: 18px 24px 48px 24px;
        box-sizing: border-box;
      }
      .so101-header-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        background: #1a1d24;
        border: 1px solid #2d3340;
        border-left: 4px solid #0071c5;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
      }
      .so101-title-block h2 {
        margin: 0 0 4px 0;
        font-size: 18px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .so101-title-block p {
        margin: 0;
        font-size: 12.5px;
        color: #94a3b8;
      }
      .so101-btn-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }
      .so101-btn {
        background: #262b36;
        color: #e2e8f0;
        border: 1px solid #3b4252;
        border-radius: 6px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s ease;
      }
      .so101-btn:hover {
        background: #323947;
        border-color: #64748b;
      }
      .so101-btn.primary {
        background: #0071c5;
        border-color: #38bdf8;
        color: #fff;
      }
      .so101-btn.primary:hover {
        background: #0284c7;
      }
      .so101-btn.accent {
        background: #ea580c;
        border-color: #fb923c;
        color: #fff;
      }
      .so101-btn.accent:hover {
        background: #f97316;
      }
      .so101-btn.success {
        background: #059669;
        border-color: #34d399;
        color: #fff;
      }
      .so101-grid-2 {
        display: grid;
        grid-template-columns: 1.15fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
      }
      .so101-grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 16px;
      }
      @media (max-width: 1280px) {
        .so101-grid-2, .so101-grid-3 {
          grid-template-columns: 1fr;
        }
      }
      .so101-card {
        background: #181b22;
        border: 1px solid #2a303c;
        border-radius: 8px;
        padding: 14px 16px;
      }
      .so101-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #262b36;
      }
      .so101-card-header h3 {
        margin: 0;
        font-size: 14px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.2px;
      }
      .so101-pill {
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
      }
      .so101-pill.warn {
        color: #fbbf24;
        border-color: #78350f;
        background: #271a0c;
      }
      .so101-pill.ok {
        color: #34d399;
        border-color: #065f46;
        background: #06241c;
      }
      .so101-cam-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
      }
      .so101-cam-box {
        background: #0f1217;
        border: 1px solid #262b36;
        border-radius: 6px;
        overflow: hidden;
      }
      .so101-cam-box img {
        width: 100%;
        height: 175px;
        object-fit: cover;
        display: block;
        background: #090b0e;
      }
      .so101-cam-meta {
        padding: 6px 8px;
        font-size: 11px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .so101-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      .so101-table th, .so101-table td {
        padding: 6px 8px;
        text-align: left;
        border-bottom: 1px solid #222731;
        font-variant-numeric: tabular-nums;
      }
      .so101-table th {
        color: #94a3b8;
        font-weight: 600;
        background: #13161c;
      }
      .so101-mono {
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
        font-size: 11.5px;
      }
      .so101-slider-row {
        display: grid;
        grid-template-columns: 125px 1fr 78px 95px;
        gap: 8px;
        align-items: center;
        margin-bottom: 7px;
        font-size: 12px;
      }
      .so101-input {
        background: #0f1217;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 4px 7px;
        font-size: 12px;
        font-family: inherit;
        width: 100%;
        box-sizing: border-box;
      }
      .so101-wp-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
      }
      .so101-wp-card {
        background: #12151b;
        border: 1px solid #282e3b;
        border-radius: 6px;
        padding: 8px 10px;
      }
      .so101-wp-card h4 {
        margin: 0 0 4px 0;
        font-size: 12px;
        color: #38bdf8;
      }
      .so101-matrix {
        background: #0d1015;
        border: 1px solid #242936;
        border-radius: 6px;
        padding: 8px;
        overflow-x: auto;
      }
    `;
    document.head.appendChild(style);
  }

  async function apiGet(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`GET ${path} failed (${res.status})`);
    return res.json();
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`POST ${path} failed: ${txt}`);
    }
    return res.json();
  }

  function downloadText(filename, text, mime) {
    const blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 150);
  }

  function ensureHeaderButtons() {
    injectStyles();
    if (!document.getElementById("so101-arch-toggle-btn")) {
      const btn = document.createElement("button");
      btn.id = "so101-arch-toggle-btn";
      btn.className = "so101-nav-btn floating-dock";
      btn.innerHTML = `<span>⚡ SO-101 Architecture</span><span class="so101-badge">No-VLA Pick &amp; Place</span>`;
      btn.onclick = () => toggleWorkbench();
      document.body.appendChild(btn);
    }
  }

  async function toggleWorkbench(forceOpen) {
    state.open = typeof forceOpen === "boolean" ? forceOpen : !state.open;
    const btn = document.getElementById("so101-arch-toggle-btn");
    if (btn) btn.classList.toggle("active", state.open);

    let overlay = document.getElementById("so101-workbench-overlay");
    if (!state.open) {
      if (overlay) overlay.style.display = "none";
      stopCamRefresh();
      stopTrajectoryPlayback();
      return;
    }

    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "so101-workbench-overlay";
      document.body.appendChild(overlay);
    }
    overlay.style.display = "block";
    overlay.innerHTML = `<div style="padding:32px;text-align:center;color:#94a3b8;">Loading SO-101 Model-Based Architecture Workbench from Physical AI Studio backend...</div>`;

    try {
      const m = window.location.pathname.match(/\/projects\/([0-9a-fA-F-]{36})/);
      const pid = m ? m[1] : "";
      state.snap = await apiGet(`/api/so101/state${pid ? `?project_id=${pid}` : ""}`);
    } catch (err) {
      overlay.innerHTML = `<div class="so101-card" style="border-color:#ef4444;color:#fca5a5;">
        <h3>Failed to load /api/so101/state</h3>
        <pre>${String(err)}</pre>
      </div>`;
      return;
    }

    try {
      renderWorkbench();
      startCamRefresh();
    } catch (renderErr) {
      overlay.innerHTML = `<div class="so101-card" style="border-color:#ef4444;color:#fca5a5;">
        <h3>UI Render Error in SO-101 Architecture Workbench</h3>
        <pre>${String(renderErr && renderErr.stack ? renderErr.stack : renderErr)}</pre>
      </div>`;
    }
  }

  function startCamRefresh() {
    stopCamRefresh();
    state.camStreamTimer = setInterval(() => {
      if (!state.open || !state.camStreamActive) return;
      const ts = Date.now();
      ["top", "plate", "wrist"].forEach((slot) => {
        const img = document.getElementById(`so101-cam-img-${slot}`);
        if (img) img.src = `/api/so101/camera/${slot}.jpg?t=${ts}`;
      });
    }, 1200);
  }

  function stopCamRefresh() {
    if (state.camStreamTimer) {
      clearInterval(state.camStreamTimer);
      state.camStreamTimer = null;
    }
  }

  function connectStudioLeaderWs() {
    if (state.leaderWs) {
      state.leaderWs.close();
      state.leaderWs = null;
      state.leaderWsConnected = false;
      renderWorkbench();
      return;
    }
    const db = (state.snap && state.snap.studio_db) || {};
    const proj = db.active_project;
    const robots = db.robots || [];
    const leader = robots.find((r) => r.type === "SO101_Leader" || r.name === "leader");
    if (!proj || !leader) {
      alert("No SO101_Leader robot found in the active Physical AI Studio project.");
      return;
    }
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/api/projects/${proj.id}/robots/${leader.id}/observations/ws?fps=15`;
    const ws = new WebSocket(wsUrl);
    state.leaderWs = ws;
    ws.onopen = () => {
      state.leaderWsConnected = true;
      updateWsStatusBadge();
    };
    ws.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === "observation" && msg.data) {
          const norm5 = MOTOR_NAMES.map((m) => Number(msg.data[`${m}.pos`] ?? 0));
          const gripPct = Number(msg.data["gripper.pos"] ?? 22);
          const res = await apiPost("/api/so101/kinematics", {
            arm: "leader",
            normalized: norm5,
            gripper_pct: gripPct,
          });
          state.snap.arms = res.arms;
          if (state.activeArm === "leader") {
            state.snap.kinematics = res.kinematics;
            state.snap.dynamics = res.dynamics;
          }
          updateLiveReadoutsOnly();
        }
      } catch (_e) {
        // Ignore transient parse error
      }
    };
    ws.onclose = () => {
      state.leaderWsConnected = false;
      state.leaderWs = null;
      updateWsStatusBadge();
    };
  }

  function updateWsStatusBadge() {
    const el = document.getElementById("so101-ws-status-pill");
    if (!el) return;
    el.className = `so101-pill ${state.leaderWsConnected ? "ok" : "warn"}`;
    el.textContent = state.leaderWsConnected
      ? "STUDIO LEADER WS LIVE (/observations/ws)"
      : "STUDIO LEADER WS STANDBY";
  }

  function renderWorkbench() {
    const overlay = document.getElementById("so101-workbench-overlay");
    if (!overlay || !state.snap) return;

    const s = state.snap;
    const db = s.studio_db || {};
    const proj = db.active_project;
    const robots = db.robots || [];
    const studioCams = db.cameras || [];
    const kin = s.kinematics;
    const dyn = s.dynamics;
    const cal = s.calibration;

    overlay.innerHTML = `
      <div class="so101-header-bar">
        <div class="so101-title-block">
          <h2>
            <span>Physical AI Studio · SO-101 Model-Based Pick &amp; Place Architecture</span>
            <span class="so101-pill ok">Zero-VLA Analytical Engine</span>
            <span class="so101-pill">Project: ${proj ? `${proj.name} (${proj.id.slice(0, 8)})` : "Repository Default"}</span>
          </h2>
          <p>
            Direct access to all 3 Studio Cameras (Top D415, Side/Plate D415, Wrist UVC) + SO-101 Leader &amp; Follower Arms (` +
      `${robots.map((r) => `${r.name}:${r.type}`).join(", ") || "follower, leader"}` +
      `) · Exact FK, 6×5 Geometric Jacobians, Published CAD Dynamics (632 g) &amp; 5-DOF Grasp IK.
          </p>
        </div>
        <div class="so101-btn-row">
          <button class="so101-btn" id="btn-so101-sync">🔄 Sync Studio DB Hardware</button>
          <button class="so101-btn accent" id="btn-so101-play-traj">${state.playingTraj ? "⏹ Stop 3D Trajectory" : "▶ Play 6-Stage Pick-Spin-Sort (3D)"}</button>
          <button class="so101-btn primary" id="btn-so101-dl-py">⬇ Python Controller (.py)</button>
          <button class="so101-btn" id="btn-so101-dl-urdf">⬇ Calibrated URDF (.urdf)</button>
          <button class="so101-btn" id="btn-so101-dl-json">📦 Full Bundle (.json)</button>
          <button class="so101-btn" id="btn-so101-close">✕ Return to Studio</button>
        </div>
      </div>

      <!-- ROW 1: ALL 3 CAMERAS + INTERACTIVE 3D KINEMATIC & WORKSPACE TWIN -->
      <div class="so101-grid-2">
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>1. Multi-Camera Station Feeds (Top D415 · Plate/Side D415 · Wrist Sonix UVC)</h3>
            <div style="display:flex;gap:8px;align-items:center;">
              <span class="so101-pill">${studioCams.length} Cameras Registered in Studio DB</span>
              <button class="so101-btn" id="btn-so101-cam-toggle">${state.camStreamActive ? "⏸ Pause Feeds" : "▶ Resume Feeds"}</button>
            </div>
          </div>
          <div class="so101-cam-grid">
            ${[
              { slot: "top", title: "Top / Overhead D415", serial: "350122060653", role: "Table & Bowl 3D Localization" },
              { slot: "plate", title: "Side / Plate D415", serial: "243622060450", role: "360° Joint-5 Spin Inspection" },
              { slot: "wrist", title: "Wrist Sonix UVC", serial: "/dev/video44 (25° cant)", role: "Close-Up Grasp Verification" },
            ]
              .map(
                (c) => `
              <div class="so101-cam-box">
                <img id="so101-cam-img-${c.slot}" src="/api/so101/camera/${c.slot}.jpg?t=${Date.now()}" alt="${c.title}" />
                <div class="so101-cam-meta">
                  <div>
                    <strong style="color:#f1f5f9;">${c.title}</strong><br/>
                    <span>${c.serial} · ${c.role}</span>
                  </div>
                </div>
              </div>
            `
              )
              .join("")}
          </div>
          <div style="margin-top:10px;font-size:11.5px;color:#94a3b8;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
            <span><strong>World Frame Convention:</strong> +X forward (toward bowls), +Y left, +Z up from table (Base Z = +29.33 mm).</span>
            <span><strong>Wrist Camera Extrinsics:</strong> 25° downward cant in <code>Body3</code> (+45 mm Y, -20 mm Z).</span>
          </div>
        </div>

        <div class="so101-card">
          <div class="so101-card-header">
            <h3>2. Interactive 3D SO-101 Kinematic Chain, COM &amp; Pick-Place Trajectory Twin</h3>
            <span class="so101-pill" id="so101-3d-phase-pill">TCP World: [${kin.tcp_world_xyz_mm.join(", ")}] mm | w5 = ${kin.manipulability_5d.toFixed(4)}</span>
          </div>
          <canvas id="so101-3d-canvas" width="620" height="245" style="width:100%;height:245px;background:#0b0e13;border-radius:6px;border:1px solid #242936;cursor:grab;"></canvas>
        </div>
      </div>

      <!-- ROW 2: LEADER & FOLLOWER ARMS + 6-STAGE PICK-AND-PLACE WAYPOINT BUILDER -->
      <div class="so101-grid-2">
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>3. SO-101 Leader &amp; Follower Arm Control + Unit Bridge</h3>
            <div style="display:flex;gap:6px;align-items:center;">
              <span id="so101-ws-status-pill" class="so101-pill ${state.leaderWsConnected ? "ok" : "warn"}">
                ${state.leaderWsConnected ? "STUDIO LEADER WS LIVE" : "STUDIO LEADER WS STANDBY"}
              </span>
              <button class="so101-btn" id="btn-so101-leader-ws">
                ${state.leaderWsConnected ? "Disconnect Leader WS" : "🔌 Stream Studio Leader Arm"}
              </button>
            </div>
          </div>

          <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;align-items:center;">
            <label style="font-size:12px;color:#94a3b8;">Active Arm:</label>
            <select id="sel-so101-arm" class="so101-input" style="width:170px;">
              <option value="follower" ${state.activeArm === "follower" ? "selected" : ""}>Follower (SO101_Follower)</option>
              <option value="leader" ${state.activeArm === "leader" ? "selected" : ""}>Leader (SO101_Leader)</option>
            </select>
            <label style="font-size:12px;color:#94a3b8;">Joint Coordinate Unit:</label>
            <select id="sel-so101-unit" class="so101-input" style="width:210px;">
              <option value="q_enc_rad" ${state.unitMode === "q_enc_rad" ? "selected" : ""}>Calibrated Encoder rad (2π/4096)</option>
              <option value="ticks" ${state.unitMode === "ticks" ? "selected" : ""}>Homed STS3215 Ticks (0..4095)</option>
              <option value="lerobot_deg" ${state.unitMode === "lerobot_deg" ? "selected" : ""}>LeRobot Degrees (360/4095)</option>
              <option value="normalized" ${state.unitMode === "normalized" ? "selected" : ""}>Studio Normalized ([-100, 100])</option>
            </select>
            <button class="so101-btn primary" id="btn-so101-map-l2f">Leader → Follower (CAD Map)</button>
          </div>

          <div id="so101-joint-sliders-container">
            ${renderJointSlidersHTML()}
          </div>

          <div style="margin-top:10px;padding-top:8px;border-top:1px solid #242936;font-size:11.5px;color:#94a3b8;">
            <strong>Leader → Follower CAD Posture Bridge:</strong>
            <span class="so101-mono" id="so101-l2f-readout">
              Follower Mapped Ticks: [${s.arms.leader_to_follower_mapping.follower_ticks.join(", ")}] |
              Gripper: ${s.arms.leader_to_follower_mapping.follower_gripper_tick} (≥${cal.safe_gripper_min_raw} safe floor)
            </span>
          </div>
        </div>

        <div class="so101-card">
          <div class="so101-card-header">
            <h3>4. 6-Stage Deterministic Pick-Inspect-Place Waypoint Builder (No VLA Needed)</h3>
            <span class="so101-pill ok">All Limits Verified: ${s.trajectory.all_within_joint_limits ? "YES" : "CHECK"}</span>
          </div>
          <div class="so101-wp-grid">
            ${WAYPOINT_ORDER.map((wpKey) => {
              const wp = s.waypoints[wpKey];
              return `
                <div class="so101-wp-card">
                  <h4>${wp.label}</h4>
                  <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">${wp.description}</div>
                  <div class="so101-mono" style="font-size:11px;color:#e2e8f0;margin-bottom:6px;">
                    q_rad: [${wp.q_enc_rad.map((v) => Number(v).toFixed(2)).join(", ")}] · Grip: ${wp.gripper_pct}%
                  </div>
                  <div style="display:flex;gap:6px;">
                    <button class="so101-btn" data-wp-load="${wpKey}" style="padding:3px 8px;font-size:11px;">Preview Pose</button>
                    <button class="so101-btn primary" data-wp-teach="${wpKey}" style="padding:3px 8px;font-size:11px;">Teach from Leader</button>
                  </div>
                </div>
              `;
            }).join("")}
          </div>

          <div style="margin-top:10px;padding-top:8px;border-top:1px solid #242936;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <strong style="font-size:12px;">KALAS Sorting Bowl Centers in World Frame (meters)</strong>
              <button class="so101-btn primary" id="btn-so101-solve-bowls" style="padding:4px 10px;font-size:11px;">Re-Solve All Waypoints via 5-DOF IK</button>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:11.5px;">
              ${["source", "good", "defective"]
                .map((bKey) => {
                  const b = s.bowls[bKey];
                  return `
                  <div style="background:#11141a;padding:6px 8px;border-radius:4px;border:1px solid #242936;">
                    <strong style="color:#f8fafc;text-transform:uppercase;">${bKey} Bowl</strong>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px;">
                      <input class="so101-input" id="bowl-${bKey}-x" type="number" step="0.005" value="${b.pos_world_m[0]}" title="X (m)" />
                      <input class="so101-input" id="bowl-${bKey}-y" type="number" step="0.005" value="${b.pos_world_m[1]}" title="Y (m)" />
                      <input class="so101-input" id="bowl-${bKey}-z" type="number" step="0.005" value="${b.pos_world_m[2]}" title="Z (m)" />
                    </div>
                  </div>
                `;
                })
                .join("")}
            </div>
          </div>
        </div>
      </div>

      <!-- ROW 3: FORWARD KINEMATICS, 6x5 JACOBIANS, 5-DOF IK SOLVER & PUBLISHED CAD DYNAMICS -->
      <div class="so101-grid-3">
        <!-- FK + 5-DOF IK -->
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>5. Forward Kinematics &amp; 5-DOF Grasp IK Solver</h3>
            <select id="sel-so101-fk-frame" class="so101-input" style="width:135px;">
              <option value="world" ${state.fkFrame === "world" ? "selected" : ""}>World Frame (+X fwd)</option>
              <option value="base" ${state.fkFrame === "base" ? "selected" : ""}>Base Frame (Body5_1)</option>
            </select>
          </div>
          <div id="so101-fk-panel">
            ${renderFKPanelHTML()}
          </div>

          <div style="margin-top:12px;padding-top:10px;border-top:1px solid #262b36;">
            <strong style="font-size:12.5px;color:#38bdf8;">Pure-NumPy 5-DOF Grasp / Placement IK Solver</strong>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:6px;">
              <div>
                <label style="font-size:11px;color:#94a3b8;">Target X (m)</label>
                <input class="so101-input" id="ik-tgt-x" type="number" step="0.005" value="${kin.tcp_world_xyz_m[0]}" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Target Y (m)</label>
                <input class="so101-input" id="ik-tgt-y" type="number" step="0.005" value="${kin.tcp_world_xyz_m[1]}" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Target Z (m)</label>
                <input class="so101-input" id="ik-tgt-z" type="number" step="0.005" value="${kin.tcp_world_xyz_m[2]}" />
              </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1.2fr;gap:6px;margin-top:6px;">
              <div>
                <label style="font-size:11px;color:#94a3b8;">Pitch Down (°)</label>
                <input class="so101-input" id="ik-pitch-deg" type="number" step="5" value="90" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Jaw Yaw (°)</label>
                <input class="so101-input" id="ik-yaw-deg" type="number" step="5" value="0" />
              </div>
              <div>
                <label style="font-size:11px;color:#94a3b8;">Save to Waypoint</label>
                <select class="so101-input" id="ik-save-wp">
                  <option value="">(Preview Only)</option>
                  ${WAYPOINT_ORDER.map((k) => `<option value="${k}">${k}</option>`).join("")}
                </select>
              </div>
            </div>
            <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;">
              <button class="so101-btn primary" id="btn-so101-solve-ik">Solve 5-DOF IK &amp; Apply</button>
              <span id="so101-ik-status" class="so101-mono" style="font-size:11px;color:#34d399;">
                ${state.lastIkResult ? `Err: ${state.lastIkResult.position_error_mm} mm (${state.lastIkResult.feasible ? "FEASIBLE" : "LIMIT"})` : "Ready"}
              </span>
            </div>
          </div>
        </div>

        <!-- 6x5 GEOMETRIC JACOBIAN -->
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>6. 6×5 Geometric Jacobian &amp; Singularity Metrics</h3>
            <select id="sel-so101-jac-mode" class="so101-input" style="width:175px;">
              <option value="J_world_rad" ${state.jacMode === "J_world_rad" ? "selected" : ""}>World J_rad (m/rad)</option>
              <option value="J_base_rad" ${state.jacMode === "J_base_rad" ? "selected" : ""}>Base J_rad (m/rad)</option>
              <option value="J_world_tick" ${state.jacMode === "J_world_tick" ? "selected" : ""}>World J_tick (m/tick)</option>
              <option value="J_world_lerobot_deg" ${state.jacMode === "J_world_lerobot_deg" ? "selected" : ""}>World J_deg (LeRobot)</option>
            </select>
          </div>
          <div id="so101-jacobian-panel">
            ${renderJacobianPanelHTML()}
          </div>
        </div>

        <!-- PUBLISHED CAD DYNAMICS & GRAVITY COMPENSATION -->
        <div class="so101-card">
          <div class="so101-card-header">
            <h3>7. Published CAD Dynamics (M(q), g(q), COM)</h3>
            <span class="so101-pill">${dyn.total_mass_g} g Total (Arm 632g + Brick ${dyn.payload_mass_g}g)</span>
          </div>
          <div id="so101-dynamics-panel">
            ${renderDynamicsPanelHTML()}
          </div>
        </div>
      </div>

      <!-- ROW 4: PER-ARM CALIBRATION & CAD ZERO ALIGNMENT WIZARD -->
      <div class="so101-card">
        <div class="so101-card-header">
          <h3>8. Any-SO-101 Arm Calibration Table &amp; Upright Reference Pose Alignment Wizard</h3>
          <span class="so101-pill">Upright Reference Pose: [0°, +93.1°, -89.5°, +1.5°, +90°] CAD</span>
        </div>
        <div class="so101-grid-2" style="margin-bottom:0;">
          <div>
            <table class="so101-table so101-mono">
              <thead>
                <tr>
                  <th>ID · Motor</th>
                  <th>CAD Joint</th>
                  <th>Range Ticks</th>
                  <th>Zero Tick</th>
                  <th>Homing Off</th>
                  <th>Limits (rad)</th>
                  <th>CAD@Zero (°)</th>
                  <th>Sign</th>
                </tr>
              </thead>
              <tbody>
                ${cal.arms[state.activeArm].joints
                  .map(
                    (j) => `
                  <tr>
                    <td>#${j.id} ${j.motor_name}</td>
                    <td>${j.cad_joint}</td>
                    <td>[${j.range_min_tick}, ${j.range_max_tick}]</td>
                    <td>${j.zero_ticks}</td>
                    <td>${j.homing_offset}</td>
                    <td>[${j.lower_enc_rad.toFixed(2)}, ${j.upper_enc_rad.toFixed(2)}]</td>
                    <td>${j.cad_at_zero_deg.toFixed(2)}°</td>
                    <td>${j.cad_sign > 0 ? "+1" : "-1"}</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
          <div>
            <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;">
              <strong>Calibrate Any New SO-101 Arm in 1 Step:</strong> Place the arm in the upright visual reference pose
              (<code>[0, 93.1, -89.5, 1.5, 90] deg</code>), enter its 5 homed STS3215 ticks below (or click <em>Use Current Ticks</em>),
              and compute <code>cad_at_zero_rad</code> + <code>cad_sign</code>:
            </div>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:8px;">
              ${kin.ticks
                .map(
                  (t, idx) => `
                <div>
                  <label style="font-size:10.5px;color:#94a3b8;">${MOTOR_NAMES[idx]}</label>
                  <input class="so101-input" id="align-tick-${idx}" type="number" value="${t}" />
                </div>
              `
                )
                .join("")}
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
              <button class="so101-btn primary" id="btn-so101-run-align">Derive CAD Zero Alignment</button>
              <button class="so101-btn" id="btn-so101-use-cur-ticks">Use Current Arm Ticks</button>
            </div>
            <pre id="so101-align-output" class="so101-matrix so101-mono" style="margin-top:8px;max-height:110px;font-size:11px;color:#93c5fd;">${
              state.lastAlignResult
                ? JSON.stringify(state.lastAlignResult.alignment_snippet, null, 2)
                : "// Click 'Derive CAD Zero Alignment' to generate kinematics_alignment.json entry for any SO-101 arm"
            }</pre>
          </div>
        </div>
      </div>
    `;

    bindWorkbenchEvents();
    draw3DCanvas();
  }

  function renderJointSlidersHTML() {
    const s = state.snap;
    const calArm = s.calibration.arms[state.activeArm];
    const kin = s.kinematics;
    const normArr = kin.normalized || [0, 0, 0, 0, 0];

    const rows = calArm.joints.map((j, idx) => {
      let minVal, maxVal, stepVal, curVal, unitSuffix;
      if (state.unitMode === "ticks") {
        minVal = j.range_min_tick;
        maxVal = j.range_max_tick;
        stepVal = 1;
        curVal = kin.ticks[idx];
        unitSuffix = "ticks";
      } else if (state.unitMode === "lerobot_deg") {
        minVal = j.lower_lerobot_deg;
        maxVal = j.upper_lerobot_deg;
        stepVal = 0.5;
        curVal = kin.lerobot_deg[idx];
        unitSuffix = "deg";
      } else if (state.unitMode === "normalized") {
        minVal = -100;
        maxVal = 100;
        stepVal = 0.5;
        curVal = normArr[idx] ?? 0;
        unitSuffix = "norm";
      } else {
        minVal = j.lower_enc_rad;
        maxVal = j.upper_enc_rad;
        stepVal = 0.01;
        curVal = kin.q_enc_rad[idx];
        unitSuffix = "rad";
      }

      return `
        <div class="so101-slider-row">
          <span class="so101-mono" style="color:#e2e8f0;">q${idx + 1} ${j.motor_name}</span>
          <input type="range" class="so101-joint-slider" data-jidx="${idx}" min="${minVal}" max="${maxVal}" step="${stepVal}" value="${curVal}" />
          <input type="number" class="so101-input so101-joint-num" data-jidx="${idx}" min="${minVal}" max="${maxVal}" step="${stepVal}" value="${Number(curVal).toFixed(state.unitMode === "ticks" ? 0 : 2)}" />
          <span class="so101-mono" style="color:#94a3b8;font-size:11px;">CAD: ${kin.q_cad_deg[idx].toFixed(1)}° (${unitSuffix})</span>
        </div>
      `;
    });

    const gripPct = s.arms.arms[state.activeArm].gripper_pct;
    const gripTick = s.arms.arms[state.activeArm].gripper_tick;
    rows.push(`
      <div class="so101-slider-row" style="margin-top:6px;padding-top:6px;border-top:1px dashed #282e3b;">
        <span class="so101-mono" style="color:#fbbf24;">#6 gripper</span>
        <input type="range" id="so101-grip-slider" min="0" max="100" step="0.5" value="${gripPct}" />
        <input type="number" class="so101-input" id="so101-grip-num" min="0" max="100" step="1" value="${gripPct}" />
        <span class="so101-mono" style="color:#fbbf24;font-size:11px;">Tick: ${gripTick}</span>
      </div>
    `);
    return rows.join("");
  }

  function renderFKPanelHTML() {
    const kin = state.snap.kinematics;
    const T = state.fkFrame === "world" ? kin.T_world_tcp : kin.T_base_tcp;
    const xyz = state.fkFrame === "world" ? kin.tcp_world_xyz_mm : kin.tcp_base_xyz_mm;
    const rpy = state.fkFrame === "world" ? kin.tcp_world_rpy_deg : kin.tcp_base_rpy_deg;

    return `
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;">
        <span><strong>TCP Position:</strong> <code class="so101-mono">[${xyz.join(", ")}] mm</code></span>
        <span><strong>RPY:</strong> <code class="so101-mono">[${rpy.join("°, ")}°]</code></span>
      </div>
      <div class="so101-matrix so101-mono">
        <table class="so101-table">
          <tbody>
            ${T.map(
              (row) =>
                `<tr>${row.map((v) => `<td>${Number(v).toFixed(4)}</td>`).join("")}</tr>`
            ).join("")}
          </tbody>
        </table>
      </div>
      <div style="margin-top:8px;">
        <strong style="font-size:11.5px;color:#94a3b8;">Link Frame Origins (World mm):</strong>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px;font-size:11px;" class="so101-mono">
          ${kin.link_chain
            .map(
              (l) => `
            <div style="background:#11141a;padding:3px 6px;border-radius:4px;">
              <span style="color:#38bdf8;">${l.link}:</span> [${l.pos_world_mm.map((v) => Math.round(v)).join(",")}]
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    `;
  }

  function renderJacobianPanelHTML() {
    const kin = state.snap.kinematics;
    const J = kin[state.jacMode] || kin.J_world_rad;
    const rowLabels = ["vx", "vy", "vz", "ωx", "ωy", "ωz"];
    const decimals = state.jacMode === "J_world_tick" ? 6 : 4;

    return `
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;">
        <span><strong>Manipulability w5(q):</strong> <code class="so101-mono" style="color:#34d399;">${kin.manipulability_5d.toFixed(5)}</code></span>
        <span><strong>Cond κ(J):</strong> <code class="so101-mono" style="color:#fbbf24;">${kin.condition_number.toFixed(2)}</code></span>
      </div>
      <div class="so101-matrix so101-mono">
        <table class="so101-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>q1 pan</th>
              <th>q2 lift</th>
              <th>q3 elbow</th>
              <th>q4 w_flex</th>
              <th>q5 w_roll</th>
            </tr>
          </thead>
          <tbody>
            ${J.map(
              (row, rIdx) => `
              <tr>
                <td style="color:#38bdf8;font-weight:700;">${rowLabels[rIdx]}</td>
                ${row.map((v) => `<td>${Number(v).toFixed(decimals)}</td>`).join("")}
              </tr>
            `
            ).join("")}
          </tbody>
        </table>
      </div>
      <div style="margin-top:8px;font-size:11.5px;color:#94a3b8;" class="so101-mono">
        <strong>Singular Values σ:</strong> [${kin.singular_values.map((v) => Number(v).toFixed(4)).join(", ")}]
      </div>
    `;
  }

  function renderDynamicsPanelHTML() {
    const dyn = state.snap.dynamics;
    const M = dyn.mass_matrix_5x5_kg_m2;

    return `
      <div style="font-size:12px;margin-bottom:6px;display:flex;justify-content:space-between;">
        <span><strong>Whole-Arm COM (World):</strong> <code class="so101-mono">[${dyn.whole_arm_com_world_m.map((v) => (v * 1000).toFixed(1)).join(", ")}] mm</code></span>
      </div>
      <table class="so101-table so101-mono">
        <thead>
          <tr>
            <th>Joint</th>
            <th>g(q) Torque (N·m)</th>
            <th>Stall Load %</th>
            <th>FF Comp (ticks)</th>
          </tr>
        </thead>
        <tbody>
          ${MOTOR_NAMES.map(
            (m, i) => `
            <tr>
              <td>q${i + 1} ${m}</td>
              <td>${dyn.gravity_torque_nm[i].toFixed(4)}</td>
              <td style="color:${dyn.servo_load_pct_of_stall[i] > 35 ? "#fbbf24" : "#34d399"};">${dyn.servo_load_pct_of_stall[i].toFixed(1)}%</td>
              <td>${dyn.feedforward_gravity_ticks[i] >= 0 ? "+" : ""}${dyn.feedforward_gravity_ticks[i]}</td>
            </tr>
          `
          ).join("")}
        </tbody>
      </table>
      <div style="margin-top:8px;">
        <strong style="font-size:11.5px;color:#94a3b8;">5×5 Joint-Space Mass Matrix M(q) (kg·m²):</strong>
        <div class="so101-matrix so101-mono" style="margin-top:4px;">
          <table class="so101-table">
            <tbody>
              ${M.map(
                (row) => `<tr>${row.map((v) => `<td>${Number(v).toFixed(5)}</td>`).join("")}</tr>`
              ).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function updateLiveReadoutsOnly() {
    const fkEl = document.getElementById("so101-fk-panel");
    const jacEl = document.getElementById("so101-jacobian-panel");
    const dynEl = document.getElementById("so101-dynamics-panel");
    const pillEl = document.getElementById("so101-3d-phase-pill");
    const l2fEl = document.getElementById("so101-l2f-readout");
    if (fkEl) fkEl.innerHTML = renderFKPanelHTML();
    if (jacEl) jacEl.innerHTML = renderJacobianPanelHTML();
    if (dynEl) dynEl.innerHTML = renderDynamicsPanelHTML();
    if (pillEl && state.snap) {
      const kin = state.snap.kinematics;
      pillEl.textContent = `TCP World: [${kin.tcp_world_xyz_mm.join(", ")}] mm | w5 = ${kin.manipulability_5d.toFixed(4)}`;
    }
    if (l2fEl && state.snap) {
      const m = state.snap.arms.leader_to_follower_mapping;
      l2fEl.textContent = `Follower Mapped Ticks: [${m.follower_ticks.join(", ")}] | Gripper: ${m.follower_gripper_tick} (≥${state.snap.calibration.safe_gripper_min_raw} safe floor)`;
    }
    draw3DCanvas();
  }

  async function pushJointUpdateFromUI() {
    const sliders = Array.from(document.querySelectorAll(".so101-joint-slider"));
    const vals = sliders.map((s) => Number(s.value));
    const gripEl = document.getElementById("so101-grip-slider");
    const gripPct = gripEl ? Number(gripEl.value) : 22;

    const payload = { arm: state.activeArm, gripper_pct: gripPct };
    payload[state.unitMode] = vals;

    try {
      const res = await apiPost("/api/so101/kinematics", payload);
      state.snap.kinematics = res.kinematics;
      state.snap.dynamics = res.dynamics;
      state.snap.arms = res.arms;
      updateLiveReadoutsOnly();
      if (state.camStreamActive) {
        ["top", "plate", "wrist"].forEach((slot) => {
          const img = document.getElementById(`so101-cam-img-${slot}`);
          if (img) img.src = `/api/so101/camera/${slot}.jpg?t=${Date.now()}`;
        });
      }
    } catch (_e) {
      // Ignore out-of-range transient drag
    }
  }

  function bindWorkbenchEvents() {
    const closeBtn = document.getElementById("btn-so101-close");
    if (closeBtn) closeBtn.onclick = () => toggleWorkbench(false);

    const syncBtn = document.getElementById("btn-so101-sync");
    if (syncBtn) {
      syncBtn.onclick = async () => {
        const m = window.location.pathname.match(/\/projects\/([0-9a-fA-F-]{36})/);
        state.snap = await apiPost("/api/so101/sync-project", { project_id: m ? m[1] : null });
        renderWorkbench();
      };
    }

    const playBtn = document.getElementById("btn-so101-play-traj");
    if (playBtn) {
      playBtn.onclick = () => {
        if (state.playingTraj) {
          stopTrajectoryPlayback();
          renderWorkbench();
        } else {
          startTrajectoryPlayback();
        }
      };
    }

    const dlPy = document.getElementById("btn-so101-dl-py");
    if (dlPy) {
      dlPy.onclick = async () => {
        const bundle = await apiGet("/api/so101/export");
        downloadText("so101_pick_place_controller.py", bundle.controller_python_script, "text/x-python");
      };
    }

    const dlUrdf = document.getElementById("btn-so101-dl-urdf");
    if (dlUrdf) {
      dlUrdf.onclick = async () => {
        const bundle = await apiGet("/api/so101/export");
        downloadText("so101-follower-calibrated.urdf", bundle.calibrated_follower_urdf, "application/xml");
      };
    }

    const dlJson = document.getElementById("btn-so101-dl-json");
    if (dlJson) {
      dlJson.onclick = async () => {
        const bundle = await apiGet("/api/so101/export");
        downloadText("so101-architecture-bundle.json", JSON.stringify(bundle, null, 2), "application/json");
      };
    }

    const camToggle = document.getElementById("btn-so101-cam-toggle");
    if (camToggle) {
      camToggle.onclick = () => {
        state.camStreamActive = !state.camStreamActive;
        camToggle.textContent = state.camStreamActive ? "⏸ Pause Feeds" : "▶ Resume Feeds";
      };
    }

    const wsBtn = document.getElementById("btn-so101-leader-ws");
    if (wsBtn) wsBtn.onclick = () => connectStudioLeaderWs();

    const armSel = document.getElementById("sel-so101-arm");
    if (armSel) {
      armSel.onchange = async (e) => {
        state.activeArm = e.target.value;
        const res = await apiPost("/api/so101/kinematics", { arm: state.activeArm });
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.arms = res.arms;
        renderWorkbench();
      };
    }

    const unitSel = document.getElementById("sel-so101-unit");
    if (unitSel) {
      unitSel.onchange = (e) => {
        state.unitMode = e.target.value;
        renderWorkbench();
      };
    }

    const l2fBtn = document.getElementById("btn-so101-map-l2f");
    if (l2fBtn) {
      l2fBtn.onclick = async () => {
        const mapped = state.snap.arms.leader_to_follower_mapping;
        state.activeArm = "follower";
        const res = await apiPost("/api/so101/kinematics", {
          arm: "follower",
          q_enc_rad: mapped.follower_q_enc_rad,
          gripper_pct: mapped.follower_gripper_pct,
        });
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.arms = res.arms;
        renderWorkbench();
      };
    }

    document.querySelectorAll(".so101-joint-slider").forEach((sl) => {
      sl.oninput = (e) => {
        const idx = e.target.getAttribute("data-jidx");
        const num = document.querySelector(`.so101-joint-num[data-jidx="${idx}"]`);
        if (num) num.value = e.target.value;
        pushJointUpdateFromUI();
      };
    });

    document.querySelectorAll(".so101-joint-num").forEach((num) => {
      num.onchange = (e) => {
        const idx = e.target.getAttribute("data-jidx");
        const sl = document.querySelector(`.so101-joint-slider[data-jidx="${idx}"]`);
        if (sl) sl.value = e.target.value;
        pushJointUpdateFromUI();
      };
    });

    const gSl = document.getElementById("so101-grip-slider");
    const gNum = document.getElementById("so101-grip-num");
    if (gSl && gNum) {
      gSl.oninput = () => {
        gNum.value = gSl.value;
        pushJointUpdateFromUI();
      };
      gNum.onchange = () => {
        gSl.value = gNum.value;
        pushJointUpdateFromUI();
      };
    }

    document.querySelectorAll("[data-wp-load]").forEach((btn) => {
      btn.onclick = async () => {
        const wpKey = btn.getAttribute("data-wp-load");
        const wp = state.snap.waypoints[wpKey];
        if (!wp) return;
        state.activeArm = "follower";
        const res = await apiPost("/api/so101/kinematics", {
          arm: "follower",
          q_enc_rad: wp.q_enc_rad,
          gripper_pct: wp.gripper_pct,
        });
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.arms = res.arms;
        renderWorkbench();
      };
    });

    document.querySelectorAll("[data-wp-teach]").forEach((btn) => {
      btn.onclick = async () => {
        const wpKey = btn.getAttribute("data-wp-teach");
        const res = await apiPost("/api/so101/teach", {
          waypoint: wpKey,
          source_arm: state.activeArm,
        });
        state.snap.waypoints = res.waypoints;
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        state.snap.trajectory = res.trajectory;
        renderWorkbench();
      };
    });

    const solveBowlsBtn = document.getElementById("btn-so101-solve-bowls");
    if (solveBowlsBtn) {
      solveBowlsBtn.onclick = async () => {
        const body = {};
        ["source", "good", "defective"].forEach((k) => {
          body[k] = [
            Number(document.getElementById(`bowl-${k}-x`).value),
            Number(document.getElementById(`bowl-${k}-y`).value),
            Number(document.getElementById(`bowl-${k}-z`).value),
          ];
        });
        const res = await apiPost("/api/so101/bowls", body);
        state.snap.bowls = res.bowls;
        state.snap.waypoints = res.waypoints;
        state.snap.trajectory = res.trajectory;
        renderWorkbench();
      };
    }

    const fkSel = document.getElementById("sel-so101-fk-frame");
    if (fkSel) {
      fkSel.onchange = (e) => {
        state.fkFrame = e.target.value;
        updateLiveReadoutsOnly();
      };
    }

    const jacSel = document.getElementById("sel-so101-jac-mode");
    if (jacSel) {
      jacSel.onchange = (e) => {
        state.jacMode = e.target.value;
        updateLiveReadoutsOnly();
      };
    }

    const solveIkBtn = document.getElementById("btn-so101-solve-ik");
    if (solveIkBtn) {
      solveIkBtn.onclick = async () => {
        const tx = Number(document.getElementById("ik-tgt-x").value);
        const ty = Number(document.getElementById("ik-tgt-y").value);
        const tz = Number(document.getElementById("ik-tgt-z").value);
        const pitch = Number(document.getElementById("ik-pitch-deg").value);
        const yaw = Number(document.getElementById("ik-yaw-deg").value);
        const wp = document.getElementById("ik-save-wp").value || null;
        const res = await apiPost("/api/so101/ik", {
          target_xyz_world: [tx, ty, tz],
          pitch_down_deg: pitch,
          jaw_yaw_deg: yaw,
          apply_to_waypoint: wp,
          update_virtual_arm: true,
        });
        state.lastIkResult = res;
        state.snap.kinematics = res.kinematics;
        state.snap.dynamics = res.dynamics;
        if (res.trajectory) state.snap.trajectory = res.trajectory;
        renderWorkbench();
      };
    }

    const useTicksBtn = document.getElementById("btn-so101-use-cur-ticks");
    if (useTicksBtn) {
      useTicksBtn.onclick = () => {
        state.snap.kinematics.ticks.forEach((t, i) => {
          const inp = document.getElementById(`align-tick-${i}`);
          if (inp) inp.value = t;
        });
      };
    }

    const runAlignBtn = document.getElementById("btn-so101-run-align");
    if (runAlignBtn) {
      runAlignBtn.onclick = async () => {
        const ticks = [0, 1, 2, 3, 4].map((i) => Number(document.getElementById(`align-tick-${i}`).value));
        const res = await apiPost("/api/so101/align", {
          arm: state.activeArm,
          measured_ticks: ticks,
        });
        state.lastAlignResult = res;
        const pre = document.getElementById("so101-align-output");
        if (pre) pre.textContent = JSON.stringify(res.alignment_snippet, null, 2);
      };
    }

    // 3D Canvas orbit controls
    const canvas = document.getElementById("so101-3d-canvas");
    if (canvas) {
      canvas.onmousedown = (e) => {
        state.dragging3d = true;
        state.lastMouse = { x: e.clientX, y: e.clientY };
      };
      window.addEventListener("mouseup", () => {
        state.dragging3d = false;
      });
      canvas.onmousemove = (e) => {
        if (!state.dragging3d) return;
        const dx = e.clientX - state.lastMouse.x;
        const dy = e.clientY - state.lastMouse.y;
        state.lastMouse = { x: e.clientX, y: e.clientY };
        state.orbitAz -= dx * 0.008;
        state.orbitEl = Math.max(0.08, Math.min(1.45, state.orbitEl + dy * 0.008));
        draw3DCanvas();
      };
      canvas.onwheel = (e) => {
        e.preventDefault();
        state.orbitDist = Math.max(0.4, Math.min(1.6, state.orbitDist + e.deltaY * 0.0008));
        draw3DCanvas();
      };
    }
  }

  function startTrajectoryPlayback() {
    stopTrajectoryPlayback();
    state.playingTraj = true;
    state.trajFrameIdx = 0;
    const playBtn = document.getElementById("btn-so101-play-traj");
    if (playBtn) playBtn.textContent = "⏹ Stop 3D Trajectory";

    state.trajTimer = setInterval(() => {
      if (!state.open || !state.playingTraj || !state.snap) {
        stopTrajectoryPlayback();
        return;
      }
      const frames = state.snap.trajectory.frames || [];
      if (!frames.length) return;
      const f = frames[state.trajFrameIdx % frames.length];
      state.trajFrameIdx = (state.trajFrameIdx + 1) % frames.length;
      const pill = document.getElementById("so101-3d-phase-pill");
      if (pill) {
        pill.textContent = `${f.phase} | w5 = ${f.manipulability_5d.toFixed(4)}`;
      }
      draw3DCanvas(f);
    }, 95);
  }

  function stopTrajectoryPlayback() {
    state.playingTraj = false;
    if (state.trajTimer) {
      clearInterval(state.trajTimer);
      state.trajTimer = null;
    }
  }

  function project3D(pt, w, h) {
    const target = [0.22, 0.0, 0.08];
    const ex = target[0] + state.orbitDist * Math.cos(state.orbitEl) * Math.cos(state.orbitAz);
    const ey = target[1] + state.orbitDist * Math.cos(state.orbitEl) * Math.sin(state.orbitAz);
    const ez = target[2] + state.orbitDist * Math.sin(state.orbitEl);

    let fx = target[0] - ex,
      fy = target[1] - ey,
      fz = target[2] - ez;
    const fn = Math.hypot(fx, fy, fz) || 1;
    fx /= fn;
    fy /= fn;
    fz /= fn;

    // Right = forward x up(0,0,1)
    let rx = fy,
      ry = -fx,
      rz = 0;
    const rn = Math.hypot(rx, ry, rz) || 1;
    rx /= rn;
    ry /= rn;

    // True up = right x forward
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    const dx = pt[0] - ex,
      dy = pt[1] - ey,
      dz = pt[2] - ez;
    const cx = dx * rx + dy * ry + dz * rz;
    const cy = dx * ux + dy * uy + dz * uz;
    const cz = dx * fx + dy * fy + dz * fz;
    if (cz <= 0.05) return null;
    const f = 430;
    return [w * 0.5 + (cx / cz) * f, h * 0.54 - (cy / cz) * f, cz];
  }

  function draw3DCanvas(overrideFrame) {
    const canvas = document.getElementById("so101-3d-canvas");
    if (!canvas || !state.snap) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // 1. Floor grid (z = 0)
    ctx.strokeStyle = "#1e2633";
    ctx.lineWidth = 1;
    for (let gx = 0.0; gx <= 0.5; gx += 0.05) {
      const p0 = project3D([gx, -0.3, 0], w, h);
      const p1 = project3D([gx, 0.3, 0], w, h);
      if (p0 && p1) {
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(p1[0], p1[1]);
        ctx.stroke();
      }
    }
    for (let gy = -0.3; gy <= 0.301; gy += 0.05) {
      const p0 = project3D([0.0, gy, 0], w, h);
      const p1 = project3D([0.5, gy, 0], w, h);
      if (p0 && p1) {
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(p1[0], p1[1]);
        ctx.stroke();
      }
    }

    // 2. World Coordinate Triad (+X red, +Y green, +Z blue)
    const o0 = project3D([0, 0, 0], w, h);
    const ox = project3D([0.08, 0, 0], w, h);
    const oy = project3D([0, 0.08, 0], w, h);
    const oz = project3D([0, 0, 0.08], w, h);
    if (o0 && ox && oy && oz) {
      [
        [ox, "#ef4444", "+X"],
        [oy, "#22c55e", "+Y"],
        [oz, "#3b82f6", "+Z"],
      ].forEach(([pt, col, lbl]) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(o0[0], o0[1]);
        ctx.lineTo(pt[0], pt[1]);
        ctx.stroke();
        ctx.fillStyle = col;
        ctx.font = "10px monospace";
        ctx.fillText(lbl, pt[0] + 3, pt[1] - 2);
      });
    }

    // 3. Draw 3 KALAS Bowls
    Object.entries(state.snap.bowls).forEach(([role, b]) => {
      const pos = b.pos_world_m;
      const rad = b.radius_m || 0.0795;
      const rgba = b.rgba || [0.8, 0.8, 0.8, 1];
      const col = `rgb(${Math.round(rgba[0] * 255)}, ${Math.round(rgba[1] * 255)}, ${Math.round(rgba[2] * 255)})`;
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= 32; i++) {
        const ang = (i / 32) * Math.PI * 2;
        const p = project3D([pos[0] + rad * Math.cos(ang), pos[1] + rad * Math.sin(ang), pos[2] + 0.012], w, h);
        if (p) {
          if (!started) {
            ctx.moveTo(p[0], p[1]);
            started = true;
          } else {
            ctx.lineTo(p[0], p[1]);
          }
        }
      }
      ctx.stroke();
      const c = project3D([pos[0], pos[1], pos[2] + 0.015], w, h);
      if (c) {
        ctx.fillStyle = col;
        ctx.font = "bold 10px sans-serif";
        ctx.fillText(role.toUpperCase(), c[0] - 18, c[1] + 3);
      }
    });

    // 4. Draw Trajectory Path
    const trajFrames = (state.snap.trajectory && state.snap.trajectory.frames) || [];
    if (trajFrames.length > 1) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let st = false;
      trajFrames.forEach((tf) => {
        const p = project3D(tf.tcp_world_xyz_m, w, h);
        if (p) {
          if (!st) {
            ctx.moveTo(p[0], p[1]);
            st = true;
          } else {
            ctx.lineTo(p[0], p[1]);
          }
        }
      });
      ctx.stroke();
    }

    // 5. Draw SO-101 Kinematic Link Chain
    const chainPts = overrideFrame
      ? [...overrideFrame.link_chain_world, overrideFrame.tcp_world_xyz_m]
      : [...state.snap.kinematics.link_chain.map((r) => r.pos_world_m), state.snap.kinematics.tcp_world_xyz_m];

    ctx.strokeStyle = "#f97316";
    ctx.lineWidth = 4;
    ctx.beginPath();
    const projChain = chainPts.map((pt) => project3D(pt, w, h));
    projChain.forEach((p, idx) => {
      if (!p) return;
      if (idx === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.stroke();

    projChain.forEach((p, idx) => {
      if (!p) return;
      ctx.fillStyle = idx === projChain.length - 1 ? "#38bdf8" : "#fde047";
      ctx.beginPath();
      ctx.arc(p[0], p[1], idx === projChain.length - 1 ? 5.5 : 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // 6. Draw Whole-Arm COM marker
    const comW = state.snap.dynamics.whole_arm_com_world_m;
    const pCom = project3D(comW, w, h);
    if (pCom) {
      ctx.fillStyle = "#ec4899";
      ctx.beginPath();
      ctx.arc(pCom[0], pCom[1], 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "10px monospace";
      ctx.fillText("COM", pCom[0] + 7, pCom[1] + 3);
    }
  }

  // Observe DOM changes in Physical AI Studio SPA to keep button mounted
  const obs = new MutationObserver(() => ensureHeaderButtons());
  if (document.body) {
    ensureHeaderButtons();
    obs.observe(document.body, { childList: true, subtree: true });
  } else {
    window.addEventListener("DOMContentLoaded", () => {
      ensureHeaderButtons();
      obs.observe(document.body, { childList: true, subtree: true });
    });
  }

  if (window.location.hash === "#so101-architecture") {
    setTimeout(() => toggleWorkbench(true), 300);
  }
})();
