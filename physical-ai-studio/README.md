<p align="center">
  <img src="docs/assets/physical_ai_studio.png" alt="Physical AI Studio" width="100%">
</p>

<div align="center">

**Physical AI Studio — Modified with the SO-101 Model-Based Pick-and-Place Architecture Workbench (Zero-VLA Edition)**

[SO-101 Architecture Workbench](#so-101-model-based-pick-and-place-architecture-workbench-no-vla-edition) • [Step-by-Step Setup for Any SO-101 Arm](#step-by-step-guide-setting-up-any-so-101-arm-without-ai-training) • [Kinematics, Jacobians & Dynamics Reference](#built-in-kinematics-6x5-jacobians-5-dof-ik--dynamics-tools) • [REST API](#so-101-architecture-rest-api-endpoints) • [Upstream Studio UI & CLI](#studio-ui)

</div>

---

## SO-101 Model-Based Pick-and-Place Architecture Workbench (No-VLA Edition)

This modified edition of Intel's **[`open-edge-platform/physical-ai-studio`](https://github.com/open-edge-platform/physical-ai-studio)** integrates our complete **analytical, model-based SO-101 Pick-Inspect-Place / Spin-and-Sort architecture** directly into the Physical AI Studio backend (`application/backend/src/so101_architecture/` & `application/backend/src/api/so101_architecture.py`) and web UI (`application/ui/src/routes/so101-architecture/` & runtime header tab **`⚡ SO-101 Architecture`**).

Instead of requiring a user to collect hundreds of demonstrations and train a Vision-Language-Action (VLA) policy, **anyone with any SO-101 arm** can open Physical AI Studio and immediately access:

1. **All 3 Camera Feeds Simultaneously**:
   - **Top / Overhead Intel RealSense D415** (`350122060653`) — Tabletop & bowl 3D block localization
   - **Side / Plate Intel RealSense D415** (`243622060450`) — Multi-view 360° Joint-5 CCW spin inspection
   - **Wrist / Gripper Sonix USB 2.0 UVC Camera** (`/dev/video44`, $25^\circ$ downward cant in `Body3`) — Close-up grasp verification
2. **Direct SO-101 Leader & Follower Arm Integration**:
   - Automatically discovers `SO101_Leader` and `SO101_Follower` calibrations (`homing_offset`, `range_min`, `range_max`, `drive_mode`) directly from Physical AI Studio's `physicalai.db` project database (`project_robots` & `project_cameras`) or live `armd` (`127.0.0.1:8765`).
   - Streams live Leader Arm observations via Physical AI Studio's native `/api/projects/{project_id}/robots/{robot_id}/observations/ws` WebSocket and maps Leader $\rightarrow$ Follower postures geometrically across independent servo calibrations.
3. **All Mathematical Tools Built In (Zero Extra Installs)**:
   - **Calibrated Forward Kinematics ($4\times 4$ $\text{SE}(3)$)** in both Base (`Body5_1`) and World (`+X` forward, `+Y` left, `+Z` up) frames.
   - **Analytical $6\times 5$ Geometric Jacobians** with respect to calibrated encoder radians ($J_{\text{rad}}$), raw STS3215 servo ticks ($J_{\text{tick}}$), and LeRobot degrees ($J_{\text{deg}}$), plus $5\times 5$ Gram manipulability $w_5(q) = \sqrt{\det(J^T J)}$, singular values, and condition number $\kappa(J)$.
   - **Published CAD Dynamics & Gravity Compensation** (`632.006 g` total across 6 rigid links + held payload mass): whole-arm Center of Mass (COM), $5\times 5$ Joint-Space Mass Matrix $M(q)$, Gravity Torque vector $g(q)$ ($\text{N}\cdot\text{m}$), % of STS3215 stall torque (`2.94 Nm`), and feedforward gravity tick compensation.
   - **Pure-NumPy 5-DOF Grasp/Placement Inverse Kinematics Solver**: Damped Gauss-Newton 5-DOF IK solver with multi-start seeding and hardware limit validation.
   - **One-Click Standalone Controller & URDF Export**: Generates a self-contained Python Pick-Inspect-Place controller (`so101_pick_place_controller.py`) and calibrated URDF (`so101-follower-calibrated.urdf`).

---

## Step-by-Step Guide: Setting Up Any SO-101 Arm Without AI Training

### Step 1 — Start Physical AI Studio
On the Intel station (with Docker Compose):
```bash
cd ~/physical-ai-studio/application/docker
docker compose --profile xpu up -d
```
Or run the standalone Studio Workbench directly from the Brickcell repository on any machine (macOS / Linux / Windows) with zero extra dependencies:
```bash
uv run python -m brickcell studio --port 7860
```
Open **<http://localhost:7860>** and click the **`⚡ SO-101 Architecture`** button in the top navigation bar (or open **<http://localhost:7860/#so101-architecture>**).

### Step 2 — Sync Your SO-101 Arms & Cameras from Physical AI Studio
1. Register your `SO101_Follower` and `SO101_Leader` arms and cameras (`top`, `side`, `wrist`) in Physical AI Studio's **Robots** and **Cameras** tabs as usual.
2. Open **`⚡ SO-101 Architecture`** and click **`🔄 Sync Studio DB Hardware`**.
   - The workbench automatically reads your project's `SO101_Follower` and `SO101_Leader` calibration dictionaries from `physicalai.db` and computes the per-joint `zero_ticks`, radian limits, LeRobot degree limits, and `SAFE_GRIPPER_MIN_RAW = 1339` anti-overload floor.

### Step 3 — Align Any New SO-101 Arm to the CAD Model (1-Step Wizard)
If you attach a brand-new SO-101 arm with different servo homing offsets:
1. Place the physical arm into the **upright visual reference pose** (`[0°, +93.1°, -89.5°, +1.5°, +90°]` in CAD coordinates: shoulder pan forward, upper arm vertical, forearm horizontal forward, wrist level, gripper roll upright).
2. In **Section 8 (Any-SO-101 Arm Calibration Table & Alignment Wizard)**, click **`Use Current Arm Ticks`** (or type the 5 homed STS3215 tick values) and click **`Derive CAD Zero Alignment`**.
3. The workbench solves:
   $$q_{\text{enc}} = (t_{\text{measured}} - t_{\text{zero}}) \cdot \frac{2\pi}{4096}, \qquad q_{\text{cad}@0} = q_{\text{cad,ref}} - s \odot q_{\text{enc}}$$
   and outputs the exact `kinematics_alignment.json` snippet for your arm.

### Step 4 — Construct Your Pick-and-Place Waypoints (IK or Leader Arm Teaching)
The deterministic architecture uses **6 canonical waypoints**:
1. `observation_home` — High-clearance stow pose keeping the top & side D415 views unobstructed
2. `source_pregrasp` — Top-down approach $75\text{ mm}$ above the source bowl center
3. `source_pick` — Precision top-down grasp at brick center height ($z = +24\text{ mm}$)
4. `plate_camera_spin` — Side D415 inspection presentation with a $28^\circ$ downward pitch and Joint-5 (`wrist_roll`) CCW sweep ($+0.45 \rightarrow +2.98\text{ rad}$)
5. `good_bowl_drop` — Verified GOOD bowl placement release pose
6. `defective_bowl_drop` — Verified DEFECTIVE bowl placement release pose

You can populate or customize these 6 waypoints in two ways:
- **Analytical 5-DOF IK (Automatic)**: Enter the $(x, y, z)$ coordinates (in World meters) of your `SOURCE`, `GOOD`, and `DEFECTIVE` bowls in **Section 4** and click **`Re-Solve All Waypoints via 5-DOF IK`**, or use **Section 5 (5-DOF Grasp IK Solver)** to solve any custom $(x, y, z, \text{pitch}, \text{yaw})$ target.
- **Physical Leader Arm Teaching**: Click **`🔌 Stream Studio Leader Arm`** to stream live joint positions from your `SO101_Leader` arm, guide the Leader Arm by hand to the desired pose, and click **`Teach from Leader`** on any waypoint card. The engine maps the Leader's encoder angles through the shared CAD frame into the Follower's calibrated encoder space automatically.

### Step 5 — Preview in 3D & Export Your Standalone Controller
1. Click **`▶ Play 6-Stage Pick-Spin-Sort (3D)`** to animate the complete trajectory in the interactive 3D workspace canvas while monitoring live manipulability $w_5(q)$ and gravity torque $g(q)$.
2. Click **`⬇ Python Controller (.py)`** and **`⬇ Calibrated URDF (.urdf)`** to download your ready-to-run standalone Python script (`so101_pick_place_controller.py`):
   ```bash
   # Dry-run kinematic & Jacobian verification (no servo motion)
   python3 so101_pick_place_controller.py --dry-run

   # Execute live Pick -> Plate-Camera Spin -> Sort on the physical SO-101 Follower arm
   python3 so101_pick_place_controller.py --execute --sort-to good
   ```

---

## Built-In Kinematics, 6×5 Jacobians, 5-DOF IK & Dynamics Tools

| Tool / Module | Mathematical Formulation | Studio UI Panel |
|---|---|---|
| **Unit Bridge** | Bridges Physical AI Studio `normalized` ($[-100, 100]$), Homed STS3215 `ticks` ($0..4095$), Calibrated Encoder `q_enc_rad` ($2\pi/4096\text{ rad/tick}$), LeRobot `degrees` ($360/4095\text{ deg/tick}$), and `so101-cad.urdf` `q_cad_rad`. | Section 3 & Section 8 |
| **Forward Kinematics** | Product of exponentials across `Revolute_1..Revolute_4` in `Body5_1` base frame and REP-103 World frame ($T_{\text{world}\leftarrow\text{base}} = R_z(+90^\circ)$ at $z_0 = +29.33\text{ mm}$), with open-mouth grasp TCP offset $\mathbf{p}_{\text{tcp}}^{\text{Body3}} = (+0.015, 0.000, -0.088)\text{ m}$. | Section 2 & Section 5 |
| **$6\times 5$ Geometric Jacobians** | $J_{\text{rad}}(q) = \begin{bmatrix} J_v(q) \\ J_\omega(q) \end{bmatrix} \in \mathbb{R}^{6\times 5}$, $J_{\text{tick}}(q) = \frac{2\pi}{4096} J_{\text{rad}}(q)$, $J_{\text{deg}}(q) = \frac{\pi}{180}\frac{4095}{4096} J_{\text{rad}}(q)$, plus 5-DOF Yoshikawa manipulability $w_5(q) = \sqrt{\det(J^T J)}$ and condition number $\kappa(J) = \sigma_{\max}/\sigma_{\min}$. | Section 6 |
| **Published CAD Dynamics** | Registered published Onshape CAD inertials (`632.006 g` across 6 links + payload mass): whole-arm COM, $5\times 5$ Mass Matrix $M(q) = \sum (m_i J_{v,i}^T J_{v,i} + J_{\omega,i}^T I_{w,i} J_{\omega,i})$, Gravity Torque $g(q) = -\sum m_i J_{v,i}^T \mathbf{g}_w$, % of STS3215 stall torque (`2.94 Nm`), and static gravity droop tick compensation. | Section 7 |
| **5-DOF Grasp IK Solver** | Pure-NumPy damped Gauss-Newton solver minimizing 3D position + antipodal-safe 2D Lambert approach-pitch + jaw-yaw residuals subject to per-arm calibration limits. | Section 4 & Section 5 |

---

## SO-101 Architecture REST API Endpoints

Mounted at `/api/so101/*` inside Physical AI Studio (`application/backend/src/api/so101_architecture.py`):

- `GET /api/so101/state?project_id=<uuid>` — Full workbench state (`studio_db`, `calibration`, `arms`, `kinematics`, `dynamics`, `bowls`, `waypoints`, `cameras`, `trajectory`)
- `POST /api/so101/sync-project` — Sync `SO101_Leader` & `SO101_Follower` calibrations and cameras from Physical AI Studio's `physicalai.db` or apply a custom calibration dictionary
- `POST /api/so101/kinematics` — Compute FK, $6\times 5$ Jacobians (`rad`, `tick`, `lerobot_deg`), manipulability, and dynamics ($M(q), g(q)$, COM) for any joint state (`q_enc_rad`, `ticks`, `lerobot_deg`, or `normalized`)
- `POST /api/so101/ik` — Solve 5-DOF Grasp/Placement IK for `target_xyz_world`, `pitch_down_deg`, `jaw_yaw_deg`
- `POST /api/so101/teach` — Capture Leader or Follower posture and save to any Pick-and-Place waypoint
- `POST /api/so101/align` — Derive `cad_at_zero_rad` and `cad_sign` from upright reference ticks `[0, 93.1, -89.5, 1.5, 90] deg`
- `POST /api/so101/bowls` — Update workspace bowl positions (`source`, `good`, `defective`) and re-solve all analytical IK waypoints
- `GET /api/so101/export` — Download standalone Python controller script, calibrated URDF, and JSON bundle
- `GET /api/so101/camera/{top,plate,wrist}.jpg` — Fetch live camera frame or annotated 3D synthetic preview

---

## Studio UI

Collect demonstrations, train policies, and run inference through a visual interface.

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/assets/readme/gui-demo.gif" alt="Physical AI Studio UI demo: configure robots and cameras, record data, train, review models, and deploy to robots" width="100%">
</p>
<!-- markdownlint-enable MD033 -->

[Start Studio with Docker](./application/docs/01-installation.md#install-with-docker-recommended) · [Native setup](./application/docs/01-installation.md#alternatively-install-natively)

<details>
<summary>Copy Docker start commands</summary>

```bash
git clone https://github.com/open-edge-platform/physical-ai-studio.git
cd physical-ai-studio/application/docker
cp .env.example .env
./setup-devices.sh --cpu # or --xpu or --cuda
docker compose up -d
```

Open <http://localhost:7860>. See the [Docker guide](./application/docker/README.md) for hardware options and setup. Configure `HF_TOKEN` in `application/docker/.env` for Hugging Face-backed policies; see [Hugging Face integration](./application/backend/docs/huggingface_integration.md).

</details>

<details>
<summary>Native development commands</summary>

In one terminal, from the repository root:

```bash
cd application/backend
uv run --extra xpu physicalai-studio serve # or --extra cpu / --extra cuda
```

In a second terminal, also from the repository root:

```bash
cd application/ui
npm install
npm run start
```

Open <http://localhost:3000>.

</details>

## Documentation

[Library](./library/README.md) · [Application](./application/README.md) · [Contributing](./CONTRIBUTING.md)
