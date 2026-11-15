# Executed validation — 13 November 2026

## Multi-Camera Dashboard View Switching, Calibrated Sim Camera Rig, Top-Only Point Cloud & Wrist Feature Analysis HUD — 14 November 2026

Updated `src/brickcell/block_inspector/real2sim.py` and `dashboard/{server.py,static/index.html,static/app.js,tests/test_server.py}` so the Tarot Card Dashboard (`:8096`) supports switching across `Top`, `Plate`, and `Wrist` camera feeds in both `Sim` and `Real` modes with calibrated 3D Real2Sim HUDs, 2D feature analysis overlays, and a strictly Top-camera 3D point cloud:

- **Calibrated multi-camera MuJoCo 4.0.1 rendering on `:18081` (`src/brickcell/block_inspector/real2sim.py`)**:
  - Replaced the static `configs/sim-scene.yaml` camera rig in `run_live_real2sim_mirror` with `live_rig` registered to `real_intel_camera_poses_world()` (`top` and `plate` D415 extrinsics/intrinsics at `640×480`) and `SONIX_T_BODY3_CAM` (`wrist` eye-in-hand camera at `640×360`, $f_x = f_y = 228.164\text{ px}$).
  - Rendered all three MuJoCo cameras (`overhead`/`top`, `side`/`plate`, and `wrist`) on every simulation tick via `_render_sim_all_pngs()`, serving `/camera/{overhead,top,side,plate,wrist}.png` and exposing `camera_top`, `camera_plate`, and `camera_wrist` in `/scene`.
- **Top-camera-only point cloud & multi-camera view switching (`dashboard/server.py`, `dashboard/static/index.html`, `dashboard/static/app.js`)**:
  - Kept default point-cloud ingestion strictly on the overhead camera (`CloudSource("top", "overhead", "http://127.0.0.1:8094/points/top")`) and removed the `Plate` cloud sensor toggle from the Depth panel.
  - Added `Top` / `Plate` / `Wrist` view selector buttons (`data-cam="top|plate|wrist"`, keys `1`/`2`/`3`) and `/frame/{real,sim}/{top,plate,wrist}.jpg` + `/stream/{real,sim}/{top,plate,wrist}.mjpg` endpoints polling `D415 Top` (`/rgb/top`), `D415 Plate` (`/rgb/plate`), and `Sonix Wrist` (`http://127.0.0.1:8091/frame.jpg` resized to `640×360`) alongside their MuJoCo counterparts.
- **Wrist & plate 2D feature analysis HUD (`analyze_camera_features`, `drawFeatureAnalysisHud`)**:
  - Implemented real-time 2D feature extraction (`Feature analysis` toggle, key `W`):
    - Detects the SO-101 purple gripper jaws (`H in [113, 146]`, `S >= 70`, `V >= 28`), extracting the left inner-jaw contour + vertical pad line (green), right inner-jaw contour + vertical pad line (cyan), and center pinch axis `u_pinch` (yellow).
    - Segments DUPLO bricks inside the aquamarine source plate via glare-immune LAB/HSV chromatic distance + distance-transform watershed + `cv2.minAreaRect`, rendering 4-corner oriented bounding polygons (red), short-axis grasp vectors (orange), and brick centroids (white).
- **Executed software & live hardware checks**:
  - `.\artifacts\arch-venv\Scripts\python.exe -m pytest dashboard/tests/test_server.py -q`: **30 passed** (including `test_multi_camera_frame_routes` and `test_analyze_camera_features_wrist_corridor_and_bricks`).
  - `.\artifacts\arch-venv\Scripts\python.exe -m pytest tests/test_station_audit.py tests/test_block_inspector.py -q`: **41 passed**.
  - `node --test dashboard/tests/test_evidence.cjs`: **6 passed**.
  - `.\artifacts\arch-venv\Scripts\ruff.exe check src tests scripts dashboard` and `.\artifacts\arch-venv\Scripts\ruff.exe format --check src tests scripts dashboard`: **passed**.
  - Deployed updated `real2sim.py` and `dashboard/` to `intel` (`192.168.50.2`), restarted `/home/intel/brickcell-sim/run-sim.sh` (`:18081`) and `/home/intel/brickcell-dashboard/run-dashboard.sh` (`:8096`), and verified live status and multi-camera endpoints.
- **Limitations**:
  - The dashboard and Real2Sim mirror remain strictly read-only with respect to hardware (`GET` only); no physical robot commands are sent by the dashboard and no real defect-classification accuracy is claimed.

## Real2Sim 6D Pose & Mesh Inertia Alignment, Top-Depth + Plate-RGB Verification, and Tarot Card Dashboard Integration — 14 November 2026

Updated `src/brickcell/block_inspector/{real2sim.py,twin.py,studio_vision.py}` and `dashboard/{server.py,static/app.js,static/index.html}` to align the live Real2Sim MuJoCo 4.0.1 scene with the physical overhead D415 depth + close-up Plate RGB cameras and expose live 3D HUD overlays on the Tarot Card Dashboard (`:8096`):

- **Executed checks on the Intel workstation (`ssh intel`, `:18081` & `:8096`)**:
  - Replaced the static `two_plate.xml` simulation service on `127.0.0.1:18081` (`/home/intel/brickcell-sim/run-sim.sh`) with the live headless `Real2SimPipeline` MuJoCo server (`python -m brickcell.block_inspector real2sim --headless --no-ssh-tunnel --live-depth-url http://127.0.0.1:8094 --armd-url http://127.0.0.1:8765 --duration-s 0`), exposing `/camera/overhead.png`, `/scene`, `/health`, and `/real2sim`.
  - Fixed runtime visual mesh orientation corruption in `SO101InspectionTwin.sync_real2sim_scene` (`src/brickcell/block_inspector/twin.py`) by composing MuJoCo's compiled mesh principal-inertia offset and quaternion (`model.mesh_pos[mesh_id]`, `model.mesh_quat[mesh_id]`) into `geom_pos` and `geom_quat` whenever `geom_dataid` is swapped dynamically, and applying a $6.5\text{ mm} / 0.18\text{ rad}$ pose hysteresis deadband with $4\times$ velocity damping to eliminate physics jitter and inter-penetration.
  - Added close-up Plate RGB (`GET /rgb/plate`) multi-view color/part verification (`_refine_bricks_with_plate_rgb`), upright vs. upside-down cavity height discrimination, side-lying stud-shelf yaw disambiguation, and pastel sorting-bowl wall subtraction in `src/brickcell/block_inspector/real2sim.py` so zero false green (`2302`) bricks are invented when no green brick is present.
  - Updated `dashboard/server.py`, `dashboard/static/app.js`, and `dashboard/static/index.html` to project and render the 3D Real2Sim HUD (8-corner wireframes, part/color/pose tags, and 64-segment KALAS plate rings) onto both the 2D live camera feed and the 3D WebGL point cloud viewer. Verified `GET http://127.0.0.1:8096/api/status` returns `ok: true`, `sim.mode: "mujoco_real2sim_live"`, and `real2sim.ok: true` while rejecting intruding human arms (`category: "human_arm"`).
- **Executed software checks (local Windows `arch-venv` & Node.js)**:
  - `.\artifacts\arch-venv\Scripts\python.exe -m pytest tests/test_block_inspector.py -q`: **17 passed** (including `test_three_plates_eight_clustered_mixed_duplo_blocks_openvino_sort` and all 4 `real2sim` tests).
  - `.\artifacts\arch-venv\Scripts\python.exe -m pytest dashboard/tests tests/test_station_audit.py -q`: **52 passed**.
  - `node --test dashboard/tests/test_evidence.cjs`: **6 passed**.
  - `.\artifacts\arch-venv\Scripts\ruff.exe check src tests scripts dashboard` and `.\artifacts\arch-venv\Scripts\ruff.exe format --check src tests scripts dashboard`: **passed**.
- **Limitations**:
  - Live Real2Sim mirroring on `:18081` and `:8096` is strictly read-only with respect to `armd` (`GET /arms/follower/state`); no physical robot motion or real defect-classification accuracy is claimed from this validation.

## Physical AI Studio SO-101 Model-Based Pick-and-Place Architecture Integration — 14 November 2026

Integrated the analytical SO-101 Pick-Inspect-Place / Spin-and-Sort architecture directly into Intel's Physical AI Studio (`open-edge-platform/physical-ai-studio`, deployed at `/home/intel/physical-ai-studio` on the Intel workstation and mirrored in `physical-ai-studio/` and `src/brickcell/studio/`):

- **Executed checks on the Intel workstation (`ssh intel`, `127.0.0.1:7860`)**:
  - Updated `/home/intel/physical-ai-studio/application/backend/src/so101_architecture/` (`engine.py`, `so101_calibration.py`, `so101_kinematics.py`, `ui_extension.js`, `SO-101.zip`, `assets/published_priors_verified.json`, `calibration/`), FastAPI router `application/backend/src/api/so101_architecture.py`, SPA index injector `application/backend/src/api/webui.py`, `application/backend/src/main.py`, React UI route `application/ui/src/routes/so101-architecture/index.tsx`, and `application/docker/docker-compose.override.yaml`.
  - Recreated and verified `physical-ai-studio-xpu` on `http://127.0.0.1:7860`:
    - `GET http://127.0.0.1:7860/api/health` returned `{"status":"healthy","restart_required":false}`.
    - `GET http://127.0.0.1:7860/` served the Physical AI Studio SPA with `<script defer src="/api/so101/ui-extension.js"></script>` injected.
    - `GET http://127.0.0.1:7860/api/so101/state` returned `studio_db.synced = true` (`/app/storage/data/physicalai.db`), automatically discovering project `white-block-picker` (`e97d7168-36bc-400a-9013-fc3c14b823f2`), `follower` (`SO101_Follower`, `/dev/ttyACM1`, serial `5B3D040991`, `unit: "normalized"`), `leader` (`SO101_Leader`, `/dev/ttyACM0`, serial `5AE7044736`, `unit: "normalized"`), and all 3 registered cameras (`top` D415 `350122060653`, `side` D415 `243622060450`, `wrist` USB2.0_CAM1 index `44`).
- **Executed software checks (local Windows `arch-venv` & Node.js)**:
  - `.\artifacts\arch-venv\Scripts\python.exe -m pytest tests/test_studio.py -q`: **5 passed** (testing `normalized` `[-100, 100]` / `ticks` / `radians` / `degrees` unit conversions, $4\times 4$ $\text{SE}(3)$ FK, $6\times 5$ Jacobians, 5-DOF damped Gauss-Newton IK convergence, published CAD mass matrix $M(q)$ and gravity torque $g(q)$, Leader $\rightarrow$ Follower CAD posture mapping, reference tick alignment derivation, `physicalai.db` auto-discovery, and `/api/so101/*` HTTP handlers).
  - `node --check physical-ai-studio/application/backend/src/so101_architecture/ui_extension.js` and `node --test dashboard/tests/test_evidence.cjs`: **6 passed**.
  - Ruff lint and format checks (`ruff check` and `ruff format --check`) passed across `src`, `tests`, `scripts`, and `dashboard`.
- **Limitations**:
  - When live camera daemons or `armd` are offline or unreachable from the container, camera feeds and arm states fall back to explicitly labeled synthetic previews (`SYNTHETIC STUDIO PREVIEW`, `studio_virtual_emulator`). No physical defect-classification accuracy or VLA policy performance is claimed.

## Multi-camera Anomalib wrapper, demonstrated physical grasp/spin parameters & depth-free 2D RGB localization — 14 November 2026

Updated `src/brickcell/block_inspector/{armd_bridge.py,cogman.py,cli.py,hardware_cell.py}` to align physical deployment with the two recorded Physical AI Studio leader-follower demonstrations (`duplo-grasp-scan-sort-v1`), multi-camera Anomalib OpenVINO CPU inference, and depth-free 2D RGB + eye-in-hand wrist visual servoing:

- **Executed software checks (local Windows `.venv` / `arch-venv`)**:
  - Extracted calibrated physical grasp widths (`18%` narrow / `24%` wide pre-grasp, `3.5%` continuous hold squeeze without relaxing during lift/spin/transit, `5.2%` empty-jaw threshold for double pick confirmation + automatic retry), side-camera presentation pose (`[0.402, -0.090, 0.146] m` at `28°` pitch, sweeping Joint 5 across `[+0.45, +2.98] rad`), and full-360° yaw IK folding across 8 drop slots (`0/16` IK slot failures in offline kinematic verification).
  - Wrapped `OpenVINODetector` (`src/brickcell/detector/runtime.py`) across `top`, `plate`, and `wrist` camera feeds (`load_anomalib_camera_detectors`, `RealCameraAnomalyInspector`), replacing the Lab dark-mark fallback whenever calibrated non-synthetic Anomalib models are present while keeping `Inspector` strictly isolated from robot commands and treating `Decision.REINSPECT` / `Decision.ERROR` as untrusted (`valid_view=False`).
  - Added depth-free 2D RGB localization (`localize_plates_and_bricks_top_rgb`) tracking live movable XY positions of all three colored KALAS plates (`source`, `good`, `defective`) and segmenting multiple DUPLO bricks via specular-glare-immune chromatic segmentation + distance-transform watershed + `minAreaRect`, plus two-stage closed-loop inner-jaw corridor visual servoing (`detect_wrist_jaw_corridor`, `servo_grasp_with_jaw_corridor`, `refine_grasp_with_wrist_rgb`, `refine_plate_drop_with_wrist_rgb`) calibrated to the physical Sonix USB wrist camera (`SONIX_T_BODY3_CAM`, `f = 228.16 px` at `640x360`), while keeping RealSense D415 depth running in parallel strictly for the Real2Sim MuJoCo 3D twin.
  - `.\artifacts\arch-venv\Scripts\pytest.exe tests/test_block_inspector.py -k "hardware_cell or anomalib_multicam" -q`: **2 passed** (`test_physical_hardware_cell_dry_run_and_safety_boundaries`, `test_anomalib_multicam_rgb_localization_and_movable_plates`).
  - `.\artifacts\arch-venv\Scripts\pytest.exe tests/test_rgbd.py tests/test_rgbd_viewer.py -q`: **18 passed**.
  - `.\artifacts\arch-venv\Scripts\ruff.exe check src tests scripts` and `.\artifacts\arch-venv\Scripts\ruff.exe format --check src tests scripts`: **passed** across all 107 files.
- **Executed live physical cell run on `intel` (`14 November 2026`, `inspection_block_0_20261114-125654.json`)**:
  - Executed `scripts/deploy_to_intel.ps1 -MaxCycles 24` against the live physical SO-101 cell (`armd :8765`, `live_depth :8094`, `focus_assist :8091`, `dashboard :8096`).
  - Completed autonomously with `succeeded: true`, `termination_reason: "parked_at_rest"`, clearing **100% of blocks from the movable aquamarine source plate** (`9` cycles completed across `duplo_2x2_3437`, `duplo_2x4_3011`, and `duplo_2x3_curved_top_2302`), placing every block cleanly inside the movable green (`good: 1`) and pink (`defective: 8`) KALAS bowls with zero blocks left on the source plate or dropped on the table (verified via post-run overhead D415 snapshot).
- **Remaining hardware item & limitations**:
  - Searching `/home/intel` confirmed no exported real Anomalib OpenVINO `model.xml` artifact has been copied to `intel` yet (`anomalib_cameras: []`), so this run exercised `inspector_mode: "lab_dark_mark_fallback"` (`dark_mark_threshold = 16.0`, where underside tube shadows during spin can exceed `16.0`); copying the teammate's calibrated `model.xml` + `.bin` + `.json` into `/home/intel/brickcell-deploy/artifacts/models/` (or referencing them in `configs/rgbd-viewer.json` / `-OpenVinoModel`) activates `anomalib_openvino_cpu` automatically without code changes.

## Live Real2Sim mirror connected to the physical Intel cell — 14 November 2026

Connected `Real2SimPipeline.run_live_real2sim_mirror` (`src/brickcell/block_inspector/real2sim.py` and `src/brickcell/block_inspector/twin.py`) to the physical Intel workstation (`intel-XPS-16-DA16260`) over a read-only SSH port-forward tunnel (`127.0.0.1:8094` -> `scripts/live_depth.py`, `127.0.0.1:8765` -> `armd`), defaulting to `--camera-mode top` (`350122060653`).

- **Live overhead D415 point-cloud + `armd` follower state mirroring**:
  - Polled `GET http://127.0.0.1:8765/arms/follower/state` (`armd_connected=true`), decoded calibrated follower joint angles (`q_enc_rad = [0.5553, -1.3629, 1.6966, -0.0767, 1.7081]`, `gripper_pct = 0.03`), and mirrored them directly into the MuJoCo 4.0.1 Desktop 3D viewer (`set_arm_state_direct`). No motion commands (`POST`) were sent to `armd`.
  - Polled live overhead D415 point clouds (`GET http://127.0.0.1:8094/points/top`, `8,738` raw points/frame, `128.6 ms` perception latency), leveled the physical wooden tabletop to `z_world = 0`, localized all 3 physical IKEA KALAS bowls (`source=[0.2978, -0.0758, 0.0]`, `good=[0.3849, +0.1004, 0.0]`, `defective=[0.2066, +0.1065, 0.0]`, all `from_prior=false`, with `19.3–44.8 mm` rim-to-rim clearance at `79.5 mm` outer bowl radius), and segmented the 8 interior DUPLO bricks across the 6 physical brick colors (`lavender`, `brown`, `yellow`, `green`, `white`, `red`) while rejecting non-brick clutter (`human_arm`, `wire`, `trash`, `robot_arm`).
- **Automated tests & linting**:
  - `.\artifacts\arch-venv\Scripts\python.exe -m pytest tests/test_block_inspector.py -k "real2sim or three_plates"`: **5 passed**.
  - `.\artifacts\arch-venv\Scripts\ruff.exe check src tests scripts` and `.\artifacts\arch-venv\Scripts\ruff.exe format --check src tests scripts` passed across all 80 files.
- **Limitations**:
  - The live Real2Sim mirror session was strictly **read-only** on the physical robot (`GET /arms/follower/state` and `GET /points/top`); no autonomous physical pick, spin, or sort motion commands were sent to the physical follower arm during this check, and no physical defect-classification accuracy claims are made.

## PatchCore RGB/depth viewer integration — 14 November 2026

Implemented `RGBDInspector` with separate modality models and explicit serial,
crop and depth-encoding contracts. RGB keeps the existing image preprocessing;
depth uses fixed 0.15–1.5 m encoding into three repeated uint8 channels, with
invalid values zeroed. Inference is gated on frame identity/freshness, aligned
shapes, RGB quality and valid depth. Synthetic models are refused; missing
models/calibration cannot produce GOOD. Both calibrated definite channels are
required for a combined per-camera decision. No cross-camera acceptance, object
correspondence, wrist integration or controller sorting was added.

Extended the Intel SDK service with `/snapshot/top` and `/snapshot/plate`,
bundling RGB JPEG and uint16 depth PNG from one aligned frameset, with frame ID,
serial, software receive time, depth scale and intrinsics. Verified live
640×480 `I;16` depth decoding from both D415s at ages 0.034/0.024 s during the
probe. Different cameras are still not hardware synchronized. Existing point,
RGB and provisional merged endpoints remain available.

Started `scripts/patchcore_depth_viewer.py` on Mac loopback port 18097 against
the forwarded SDK service on 18094. Verified both API paths and RGB/depth input
previews, explicit REINSPECT with missing crop, crop application, and plate
valid-depth fraction 0.9946 for the diagnostic crop `[320,30,410,125]` (not a
calibrated object crop). No trained real RGB/depth model was configured; only a
synthetic exported local model was found and was not used. Real model inference,
depth training, threshold calibration and real defect accuracy remain untested.

Executed `uv sync --locked`, `uv run --locked pytest -q`: **94 passed, 5 optional
integration tests skipped**; Ruff lint/format checks passed for `src tests
scripts`. New injected tests cover distinct modality inputs, fixed encoding,
synthetic/contract rejection, missing thresholds, stale frames, missing depth,
camera mismatch, decisions and lossless depth snapshot decoding. Live viewer
JavaScript parsed successfully and the HTML was opened in the Mac browser;
automated visual interaction remains unavailable. All operations used CPU/core
dependencies on Mac and the separate existing SDK environment on Intel.

## Brick-002 opposite side and collection pause — 14 November 2026

Saved fresh RGB previews/subsampled point packets under ignored
`artifacts/brick-002-other-side-review/`. The plate view shows studs in profile
on the right, versus left in the prior side view, supporting a distinct opposite
side observation. Both views are unobstructed. Recorded the same object/session
and source hashes, with no model decision, partition or camera transform.
Five clean poses are now visually reviewed for brick-002 (studs up, studs
toward plate, hollow underside, side, opposite side); this is not verified
complete geometry or five independent objects. Updated all existing review
pages to a saved-photo gallery and next-brick studs-up diagram. No core source
changed.

## Brick-002 accepted narrow side — 14 November 2026

Saved fresh RGB previews and subsampled point packets under ignored
`artifacts/brick-002-side-review/`. Visual review confirms the upright blue
brick's narrow curved side faces the plate camera, with studs in profile and
no hand occlusion. Retained the same brick-002/session group and source hashes;
assigned no dataset partition or model decision and applied no old camera
transform. Updated the existing HTML guide and review pages with the saved
photo and an opposite-side target requiring a 180-degree turn. No core source
changed during capture.

## Brick-002 accepted back and next-pose guide — 14 November 2026

Saved fresh RGB previews and subsampled point packets under ignored
`artifacts/brick-002-back-review/`. Visual review confirms the blue brick's
hollow underside and circular tubes facing the plate camera, unobstructed.
Corrected the guide's simplified smooth-back wording to match the actual
underside. Retained brick-002 and its existing session group, with no model
decision, partition or cross-camera transform applied. Updated the existing
HTML pages to show this saved photo and a 90-degree clockwise side-view target.
The served guide returned HTTP 200; no core source changed.

## Brick-002 orientation guide and accepted studs-facing view — 14 November 2026

Captured RGB previews and subsampled point packets under ignored
`artifacts/brick-002-orientation-check/`. Plate RGB review confirms all four
studs facing the camera, without hand occlusion. Recorded this as brick-002's
human-labeled clean studs-facing pose, in the existing session group, with no
partition or detector decision. Created a local HTML guide containing native
2D SVG orientation diagrams, the actual accepted photo, a live plate preview
and explicit half-turn instructions for the next back view. The diagrams are
schematic, not measured brick geometry. Guide HTML returned HTTP 200 and inline
JavaScript syntax passed; opened it in the Mac browser. No core source changed.

## Brick-002 repeat collection review — 14 November 2026

Repeat-02 showed an empty board in both views and was excluded from the clean
brick collection. Repeat-03, saved under ignored
`artifacts/brick-002-repeat-03-review/`, visibly shows the blue brick lying flat
with studs upward. Recorded it as a clean repeat, not the requested studs-
toward-plate pose or additional surface coverage. Saved source hashes and point
receive times (0.121 s skew), retaining object `brick-002` and the same capture
session group. No model prediction, dataset split or camera transform was
applied. No source code changed for these capture reviews.

## Confirmed clean brick-002 review capture — 14 November 2026

After the operator confirmed a new clean physical brick, saved fresh top/plate
JPEG previews and subsampled SDK clouds to ignored
`artifacts/brick-002-clean-front-review/`. Visual review shows a blue square
four-stud brick lying flat, with the red brick-001 outside the board. Recorded
object `brick-002`, session `brick-002-clean-20261114-a`, human GOOD label,
source SHA-256 hashes, camera serials and source timestamps in a capture record.
No model decision or dataset partition was assigned. Both image views are
unobstructed. RGB previews and point packets were requested separately; point
receive timestamps differ by 1.67 s, not hardware synchronization. JPEG previews
are not native lossless camera captures, and clouds are subsampled rather than
full depth images. The plate framing differs from earlier calibration, so no
old camera transform was applied. No source code changed for this collection.

## Extensive saved-scan reconstruction audit — 14 November 2026

Implemented offline geometry diagnostics in `brickcell.reconstruction` and
executed `scripts/reconstruction_audit.py` on the five saved brick-001 pose
clouds. Preserved raw captures and hashes; retained exact Python/package
versions, cleanup counts, fitted transforms, random seed and split counts in
ignored `artifacts/brick-001-reconstruction-audit/audit-report.json`.

Compared largest-connected-cluster cleanup (9 mm DBSCAN radius, three-point
minimum) for two-camera clouds and plate-only clouds. Cleanup can discard real
disconnected surfaces; counts are reported rather than treated as ground truth.
Aligned each pose against the upright reference using 24 proper PCA rotation
seeds and up to 50 trimmed ICP iterations. Fitting uses 80% of source and target
points, with the other 20% withheld from fitting. Candidate selection uses
bidirectional fitting distances; held-out point distances are reported
separately. This random point holdout is NOT independent object, session or
pose validation, and no inspection dataset split was created or changed.

Both-camera held-out symmetric mean distances were 8.57 mm narrow, 10.18 mm
back, 9.29 mm repeat underside and 9.32 mm initial lying pose. Plate-only
distances were 8.72, 6.30, 5.71 and 6.00 mm respectively. Plate-only performs
better on three of four fits but not the narrow pose. Similar scores from
different seeds demonstrate unresolved orientation ambiguity. These distances
are sparse nearest-neighbor interpolation diagnostics, not physical accuracy.

Exported cleaned source clouds, per-pose aligned clouds, two exploratory
composites, comparison plots and an interactive HTML review. The new composites
contain 1,302 two-camera or 869 plate-only points, excluding the repeated
underside from their coverage. All results remain `accepted: false`; no solid
mesh, missing surface, calibrated defect result or automatic robot action is
inferred. The plate-only view avoids the approximate cross-camera transform
within each source scan but does not resolve cross-pose registration ambiguity.

Added tests for proper rigid transforms, isolated outlier removal, invalid
coordinates, bidirectional partial-surface metrics and recovery of a known
asymmetric pose with fitting/holdout accounting. All five focused tests passed.
The report/UI distinguish real scans from unverified registration. Viewer HTML
syntax and served assets are checked separately; browser automation is
unavailable, so rendered interaction has not been automated.

Final checks for this change: **85 passed, 5 optional integration tests skipped**;
Ruff lint and format checks on `src tests scripts`, and diff whitespace checks
passed. The generated viewer JavaScript parsed successfully; HTML, JSON report,
three selectable PLY assets and both plotted comparisons returned HTTP 200
from the loopback review server. Opened the new HTML in the Mac default browser.

## Brick-001 repeat underside and multi-pose candidate — 14 November 2026

Captured `artifacts/brick-001-other-side-review/` following the operator's
other-side cue. RGB review shows the broad underside again; recorded pose
`underside-repeat-operator-other-side` rather than claiming opposite-side
coverage. Existing extraction retained 129 top and 285 plate points.

Added and executed `scripts/align_pose_preview.py` on upright, narrow, back
and repeat-underside rough clouds. Tested 24 proper PCA-axis seeds per pair,
30 iterations of trimmed nearest-neighbor ICP against the first pose, retaining
80% of correspondences for fitting. Best trimmed fitting distances were 4.21,
4.37 and 3.91 mm; alternative seeds have similar scores. These are in-sample
fit measures, not physical accuracy, and the visible cloud remains noisy.
Saved candidate PLY, source-pose-colored plot and rigid transforms under ignored
`artifacts/brick-001-composite-review/`, marked `accepted: false`. No complete
mesh, verified hidden surfaces, calibrated geometry or trained model is claimed.
Raw pose clouds remain separate and preserved for later reconstruction work.

## Brick-001 back/underside pose — 14 November 2026

Saved RGB and SDK point packets under ignored `artifacts/brick-001-back-review/`.
RGB review shows the broad underside with circular tubes facing the plate
camera, consistent with the operator's back-view label. Processed top ROI
`[242,211,278,271]` and plate ROI `[360,59,452,136]` using the existing red
selection, radius filter and provisional transform. Exported a rough two-view
cloud, plot and pose report. Kept this pose separate from earlier orientations;
the cloud does not imply measured cavity geometry, complete coverage or a
cross-pose reconstruction. No source code changed; processing and diff
whitespace checks completed successfully.

## Brick-001 narrow-side pose — 14 November 2026

Saved fresh RGB/SDK point packets in ignored `artifacts/brick-001-narrow-review/`.
Visual review shows the narrow face obliquely toward the plate camera, with
part of the adjacent long face visible. Software receive timestamps differ by
0.121 s; this is not hardware synchronization. Reused the provisional camera
transform and capture-specific ROIs: top `[224,216,280,258]`, plate
`[379,62,441,143]`. Red selection plus the 60 mm review radius retained 129 top
and 147 plate points. Exported a two-camera rough PLY, source-colored preview
and report under the same folder. This pose remains separate from previous
object poses; no cross-pose reconstruction, training or defect test occurred.
No source code changed; existing processing completed successfully and diff
whitespace validation passed.

## Brick-001 upright pose — 14 November 2026

Captured RGB/point packets under `artifacts/brick-001-rotated-review/`. RGB
review shows the same red brick standing upright with studs toward the plate
camera. Kept this distinct pose separate from the earlier lying-down capture;
camera registration does not register a rotated object into its previous pose.
Exported a rough two-camera upright PLY and preview using the existing review
transform. Added configurable ROI/pose fields to `merge_brick_preview.py` and
a review-only 60 mm median-centered radius to remove distant stray depths.
This filter assumes a small brick and may discard valid surfaces; raw packets
are preserved. No new camera calibration, training or defect decision occurred.

## Brick-001 coarse merged preview — 14 November 2026

At the operator's request to proceed with coarse visualization, captured
brick-001 RGB and point packets from both running live endpoints under ignored
`artifacts/brick-001-merged-review/`. Applied the latest provisional transform
without refitting. Capture-specific pixel ROIs and red-color selection retained
136 top and 195 plate points; exported `brick-001-rough-merged.ply`, a source-
colored plot and report. Sparse visible surfaces and stray depth points remain;
this is not a complete solid reconstruction or defect measurement.

Added optional `--transform` to the live review server, with camera-serial and
freshness checks for a third merged panel. Reports retain `provisional: true`
and software frame skew. The two cameras are not hardware synchronized; keep
the object still. Coarse review transforms are not used by robot/detector code.

## Clearer labelled-board plate view — 14 November 2026

Retrieved RGB and point packets in `artifacts/tilted-board-review/` after the
operator changed the setup. The board appears flat and the plate perspective
is higher, so do not assume the previously estimated camera extrinsics remain
valid. Used visually reviewed pattern-boundary pixels in A/B/C/D order;
corrected the initial top selections to exclude the gray tape border.
Provisional corner residuals are 15.78, 4.31, 7.73 and 6.48 mm. Top plane RMS
is 1.35 mm; plate plane RMS is 0.84 mm. Saved input corner JSON, packets,
`provisional-transform.json` (`accepted: false`) and a provisional scene PLY.
This is improved coarse registration, not independent validation or calibrated
millimetre defect measurement. No live transform was applied. No source code
changed during this capture/analysis run.

## Labelled board registration review — 14 November 2026

Retrieved fresh RGB/point packets for the A/B/C/D labelled board under ignored
`artifacts/labelled-board-review/`. Labels permit manual verification of the
physical corner order. Extended the pilot script to accept explicit directory
and corresponding pixel JSON, validate shapes, and retain pixels/SDK serials.
Refitting yielded corner residuals 12.47, 22.14, 6.57 and 35.46 mm; registration
remains `accepted: false`. Estimated board edge lengths differ between cameras
by up to about 36 mm despite small plane residuals (1.31 mm top, 0.74 mm plate).
Small plane residuals alone therefore do not establish metric registration.
No cause was isolated; manual pixel selection, grazing view, uncorrected
distortion and depth error remain possible contributors. The new provisional
cloud is for review only; no live or robot transform was applied. A tilted
board observation is the next planned check.

## Provisional orange-board outline registration — 14 November 2026

After the operator repositioned the board, both full outlines were visible.
Fresh RGB and SDK point packets were saved in ignored
`artifacts/orange-corner-review-02/`. SIFT with a 0.7 ratio test found zero
matches. Testing the eight possible outline orientations with normalized
red-green image correlation produced a best coefficient of 0.664 versus
0.155 or lower for alternatives; this is supporting evidence, not calibration.

Executed capture-specific `scripts/board_alignment_pilot.py`: manually chosen
corresponding outline pixels, board depth-plane RANSAC at 3 mm, pinhole ray-plane
intersection, then a rigid transform. Top plane: 1,415/1,508 inliers, 1.36 mm
RMS. Plate plane: 2,300/2,306 inliers, 0.63 mm RMS. Corresponding corner residuals
are 15.98, 8.03, 14.57 and 9.86 mm. Exported `provisional-transform.json` with
`accepted: false` and `provisional-merged-scene.ply`. This is a rough scene
alignment only, with no independent validation, distortion correction or
defect-scale accuracy. It is not applied to the live viewer or robot control.

## Orange-board color corner candidates — 14 November 2026

Fetched fresh RGB frames from both live-depth review endpoints and executed
`scripts/orange_corners.py` on explicit manually reviewed target polygons.
Normalized `(red-green)/(red+green+blue+1)` contrast plus Shi-Tomasi detected
49 top-view and 64 plate-view corner candidates. Saved numbered overlays,
contrast channels and candidate pixel coordinates under ignored
`artifacts/orange-corner-review/`. Visual review shows pattern intersections
and extra edge candidates. The plate image clips the right target boundary.
These are unordered candidates, not cross-camera correspondences or evidence
of a successful calibration; no transform or merged cloud was generated.
The polygons are capture-specific. An irregular physical pattern can provide
shared landmarks if correspondences are verified; standard equal-square
checkerboard object coordinates are inappropriate for this target.

## Live D415 point-cloud review — 14 November 2026

Added and deployed `scripts/live_depth.py` as a separate Intel SDK process on
loopback port 8094, forwarded to Mac loopback port 18094. It owns both D415s
while the RGB bench on 8092 is stopped; the wrist helper and teammate UI are
untouched. Each camera produces aligned RGB/depth at requested 640×480/15 fps;
the review API subsamples every fourth pixel and retains depths 0.15–1.5 m.
SDK point coordinates are displayed in separate rotatable canvas views; they
are not registered or merged. Packets include camera serial, intrinsics, depth
scale and software timestamps; absent/stale (>3 s) packets return HTTP 503.
No robot or detector integration exists in this viewer. Browser visual layout
and interaction have not been verified by automation.

The new physical orange/gray checkerboard was visible in both RGB previews,
but exhaustive OpenCV SB searches for 3–8 inner corners per dimension detected
no full pattern in either view. Square counts/size are pending operator input.
No camera transform was estimated. The board appears to have uneven squares;
this observation alone does not establish the cause of failed detection.

## Brick-001 matched raised/clean experiment — 14 November 2026

Captured the operator-confirmed raised change and then its clean reference
with plate SDK serial `243622060450`, 640×480 aligned RGB/depth. Captures are
`brick-001-raised-DGMCha` and `brick-001-matched-clean-GhBs4I`, retained on Intel
under `/home/intel/brickcell-captures/` and locally in ignored artifacts with
SDK metadata and PLYs. RGB review shows gray material among the studs in the
first and its absence in the second; placement appears closely matched.

Direct raw-depth comparison in manually selected ROI `[410,302,444,324]`
has 432 paired valid pixels (57.8%); median signed difference is 0 mm and
95th-percentile absolute difference is approximately 6 mm. Front control ROI
`[385,344,439,370]` has 947 paired valid pixels (67.5%), median signed difference
0 mm, and 95th-percentile absolute difference 1 mm. This is an uncalibrated
diagnostic, with no registration, repeated noise baseline, defect threshold or
accuracy claim. Missing depth and pose variation remain confounders. Keep both
captures grouped as the same physical `brick-001` in any future partitioning.
Restored the RGB bench and verified its HTTP status endpoint after capture.

## Brick-region 3D extraction — 14 November 2026

Executed `scripts/compare_depth_scans.py` on the existing clean and later
operator-labeled defective plate captures with manually reviewed pixel ROI
`382 262 437 311`. Exported separate vertex-only PLYs, RGB crops, a static 3D
comparison and a report retaining source PLY hashes under ignored artifacts.
The clean ROI contains 1,985 points with 73.65% usable depth pixels; the later
ROI contains 2,139 points with 79.37% usable depth pixels. RGB review shows
the studs and later added material at the top. These are partial visible-surface
scans; ROI edges can contain background. No table-plane segmentation or pose
registration was performed. Intrinsics are fixed to the recorded plate SDK
640×480 probe; the parser supports that SDK PLY layout only.

Correction to the earlier conversational conclusion: a zero median depth
difference across a broad image region does NOT establish that a defect is
appearance-only. Missing depths, background pixels, pose changes and sensor
noise must be handled before attributing any difference to a defect. No depth
defect threshold or accuracy has been measured. Physical object/session identity
for these older generic clean/defective folders is unconfirmed, so these are
review artifacts and must not enter formal train/val/test partitions yet.

Software verification: `uv run --locked pytest -q` passed 80 tests with five
optional integration tests skipped; Ruff lint and format checks passed on
`src tests scripts`. No new camera or robot access occurred for this extraction.

## RealSense depth probe — 14 November 2026

Stopped the temporary RGB bench so it released both D415 devices, then ran the
hardware-only `scripts/realsense_depth_capture.py` on Intel with transient
`pyrealsense2` 2.59.1. SDK serials `350122060653` (overhead/top placement) and
`243622060450` (plate/side placement) both produced aligned 640×480 RGB and
16-bit depth frames, 1 mm depth scale metadata, human-viewable depth previews,
and RGB-colored PLY point clouds. The captures were real and non-synthetic;
depth was not passed into PatchCore or used for a defect decision. The SDK
serials differ from the earlier V4L2 by-id labels, so future hardware runs must
enumerate and record the SDK identity. The RGB bench must be restarted after
the depth process exits.

## Timed five-face capture — 14 November 2026

Ran the operator-assisted sequence on Intel for `red-brick-001`, session
`clean-sequence-001`, with faces `front right back left top` at 30-second
intervals using the side D415. All five fresh bursts completed and the sequence
record was written at `/tmp/brickcell-multicam/data/face-capture-runs/` on Intel.
Each result was REINSPECT because no real defect model/calibration is configured;
this is expected. Face labels are operator supplied and require image review.
No robot movement or automatic defect decision occurred.

Added `scripts/capture_faces.py`, a timed operator-assisted face sequence. It
requires explicit object/session IDs, face order, camera and label, captures at
the configured interval (default 30 seconds), and never commands the robot or
makes defect decisions.

## Real-image reference-only PatchCore pilot — 14 November 2026

Taped capture `21d2b3935a874fb4b60a229b5a70dc00` clearly shows the gray patch.
The frozen `red-fixed-v1` checkpoint scored it **2.485419**, versus fitting
reference **2.230126** and clean repeat **2.273086**. The brick was turned
upright and partly clipped relative to the lying-down fitting pose. This
confounds pose/framing and patch effects; no defect-specific accuracy or
calibrated decision is claimed. Maps and comparison JSON were saved.

After recentering, visually verified clean capture
`1c47f4b1c90d4892af2d20705b5bab9b` at crop `[280,360,365,465]` (85×105).
Fitted a separate clean-only reference checkpoint at
`artifacts/pilots/red-fixed-v1/`, with explicit object/session/crop provenance.
Scored its fitting image **2.230126** and a fresh same-pose clean repeat
**2.273086**. Visually verified both contain the clean brick. This is a
same-object repeatability diagnostic, not validation or test accuracy. A taped
capture matching this new camera pose is still needed. Tests remained
**80 passed / 5 optional integration tests skipped**, with Ruff checks passing.

Clean recapture `39dbc2ac3ab1440397e6ed270440176b` failed visual review:
the camera viewpoint had changed, the crop contained tabletop and the brick
was partially outside the full-frame right edge. Moved the image to review
locally and remotely, excluded it from training, and did not fit a new model.
The taped lying-down capture still lacks a matched clean reference.

Capture `cc5b203826114bf780baf55b033d7f4b` visibly shows a rectangular gray
patch on the lying-down red brick's front face. Saved and visually checked the
80×95 raw crop `[265,350,345,445]`. This pose/crop differs from the original
upright clean reference; a matched clean capture is required before attributing
any score difference to the patch. No new accuracy result was claimed.

Follow-up capture `e8f03aa228bc421bbf0938aac35df303` scored **2.327533**
with the same frozen reference checkpoint. The brick had rotated relative to
the clean reference and a dark mark was still not visually identifiable. Kept
this image in the review collection, not verified defective data. No threshold
or defect-detection success is inferred from its distance.

Added `scripts/appearance_pilot.py` for an explicitly uncalibrated same-object
appearance experiment, separate from the formal train/val/test workflow. Fitted
Anomalib 2.7.0 PatchCore on the real clean red-brick crop using CPU, ResNet-18,
128×128 input, layer2/layer3, seed 42 and 5% coreset. One fitting reference
produced 11 coreset features. Saved the checkpoint hash, image hashes, versions
and no threshold in ignored `artifacts/pilots/red-reference-v1/`.

Reloaded the checkpoint and executed real image inference: fitting reference
distance **2.060604**, later unverified capture **2.407176**. Saved anomaly maps
and overlays. No dark mark was visually identifiable in the later capture, so
it was moved from the labeled defective collection into a review collection.
Its report retains the human label and adds an UNVERIFIED review note. It is
not a verified defective test image. Pose/lighting variation can explain score
changes; these two scores establish neither defect detection nor accuracy.

This pilot has no calibration, no held-out evaluation, and no GOOD/DEFECTIVE
decision or robot integration. Both captures retain the same physical object
ID; they must not be split across formal train/val/test partitions. The normal
dataset validation and production calibration paths were not weakened.

Executed `uv run --locked pytest -q`: **80 passed, 5 optional integration tests
skipped**; `uv run --locked ruff check src tests scripts` and format check passed.
The new pilot additionally completed actual fit/checkpoint-reload/inference.

## First real brick capture — 14 November 2026

After the operator placed a clean blue brick in the side D415 view, visually
verified it and set crop `[270,310,366,434]` in the 640×480 source. Added scoped
single-camera collection without relaxing three-view acceptance requirements.
Captured a fresh side-only burst and saved one unannotated 96×124 PNG as
`blue-brick-001` / `side-session-001`, human label GOOD. Retrieved and visually
verified the saved crop. Maximum adjacent-frame motion was 2.54 grayscale levels.
The detector result remained REINSPECT because no real calibration exists.

Executed tests: **80 passed, 5 optional integration tests skipped**. Ruff lint,
format and diff whitespace checks passed. This is one human-labeled example,
not a trained/evaluated dataset or evidence of real defect accuracy. The object
ID must remain unchanged if this same brick is later marked.

## Three-camera bench and burst fusion — 14 November 2026

Deployed the loopback-only bench on Intel port 8092, forwarded locally to 18092.
Both D415 feeds and the upright wrist feed returned fresh images. D415s reported
640×480; the wrist source reported 1920×1080. Each preview JPEG endpoint returned
HTTP 200 and JPEG bytes. Added a raw `/frame.jpg` endpoint to the existing wrist
helper so focus text/rectangles do not enter collected images; increased its
focus median window from 7 to 15 samples. No arm service was accessed.

The bench reports a 15-frame median focus measure and median absolute deviation
as jitter, using light Gaussian filtering for focus measurement only. One final
status sample reported jitter of 2.06% overhead, 13.92% side and 1.27% wrist.
These are scene-dependent diagnostics, not measured defect accuracy or a claim
that sensor noise has been eliminated.

Added object-scoped 10-frame fresh bursts, explicit crop/operator-confirmation
requirements, capture timestamps, a provisional motion gate and per-camera
calibrated decision fusion. Unit tests verify isolated spikes, sustained defects,
missing/uncalibrated views, wrong/duplicate identities, same-batch RGB capture,
and that a human GOOD label cannot become an uncalibrated GOOD prediction.
The deployed service refused an unconfirmed capture with HTTP 400.

Executed `.venv/bin/python -m pytest -q`: **79 passed, 5 optional integration tests
skipped**. Ruff lint and format checks on `src tests scripts` passed. Browser
automation was unavailable, so visual browser layout was not verified; live HTTP
status and all three image endpoints were verified directly.

No real defect model is configured. Real-model burst inference, calibration of
the temporal/motion rules and real three-view defect accuracy remain untested.
The combined live result cannot be GOOD until all required models/views qualify.
Camera timestamps are receive times, not hardware synchronization, and object
correspondence still requires operator confirmation. No training photos were
captured from an unidentified scene during this verification. See `MULTICAMERA.md`.

## Intel camera connectivity — 14 November 2026

Connected to the pinned Intel SSH host and enumerated USB/video devices. Both
D415s (`318123021715`, `344223025431`) returned 640×480 BGR frames through
OpenCV/V4L2 using stable by-id paths; JPEGs were saved for visual review. The
Sonix device was busy with an existing focus-assistance process. Its metrics
reported 1920×1080 at 30 fps and one annotated JPEG was retrieved through the
existing loopback MJPEG endpoint, without opening the device again. These
snapshots were copied to ignored `artifacts/camera-check/brickcell-camera-check/`.

Visual review confirmed overhead, side and upside-down wrist views respectively.
No brick defect classification, depth capture, camera calibration, real-image
training or robot movement was performed. No remote packages/services were
installed or stopped. A local SSH tunnel exposes the existing focus preview at
`127.0.0.1:18091`; its focus score is not a defect score.

## Real-camera capture tooling — 14 November 2026

Added explicit OpenCV camera capture with BGR-to-RGB conversion, checked pixel
crops, unique PNGs, object/session/view identity and quality sidecars. Optional
inspection uses the existing service and refuses synthetic-trained models.
Added `configs/patchcore-real.yaml` and `docs/REAL_CAMERA.md` for collection,
partitioning, real-image training and new-capture inspection.

Executed `.venv/bin/python -m pytest -q`: **73 passed, 5 skipped** (four optional
real-model tests and one live-simulator test). Ruff lint and format checks on
`src tests scripts` passed. The capture module's `--help` ran successfully.
Camera tests use injected frames: RGB channel/crop correctness, failure cleanup,
dataset path validation and capture-to-manifest identity preservation.

No physical camera capture or real-image training was performed. An attempted
read-only station check failed because the `intel` SSH alias did not resolve;
no remote configuration was changed. Camera device selection, OS permissions,
focus/exposure, capture blocking behavior and quality thresholds require checks
on the selected hardware. The capture utility is on demand, not a bounded-latency
robot controller camera adapter. Existing simulated accuracy is not evidence of
real defect accuracy.

All results below were produced locally in this repository. No remote Intel server, SSH login, real camera, physical arm, or cloud training was used during implementation.

## Host and versions

| Item | Observed value |
| --- | --- |
| CPU | Apple M4 Max (`sysctl -n machdep.cpu.brand_string`) |
| OS / architecture | macOS 15.6 / native arm64 |
| Python | 3.12.11 (uv-managed ARM64) |
| uv | 0.10.6 |
| Anomalib | **2.7.0** |
| torch / torchvision | **2.10.0 / 0.25.0** |
| OpenVINO | **2026.4.1**, runtime build `2026.4.1-22982-e213a147257-releases/2026/4` |
| OpenVINO devices | **`['CPU']`** |
| PyTorch MPS / CUDA | Available / unavailable; training explicitly CPU |
| OpenCV / Pillow | 5.0.0.93 / 13.0.0; conversion probe passed |

`artifacts/doctor.json` contains actual probes. `uv.lock` resolves 89 packages across target platforms; 87 were installed/audited on this Mac. `uv sync --locked --offline` succeeded after initial installation. No CUDA/XPU/NPU packages were added. Linux x86_64 resolution exists, but installation/execution on Linux has not been tested.

## Executed outcomes

- Dataset generator and validator succeeded: 12 good train, 6 good + 6 defective val, 6 good + 6 defective test. All 36 images are explicitly synthetic. Duplicate/missing/corrupt/leakage cases are tested separately.
- Actual Anomalib `Patchcore` / `Engine.fit` succeeded on CPU using official pretrained `resnet18.tv_in1k`. Fit + calibration took about **6.96 seconds** after imports. This small-fixture timing is not a real-dataset training estimate.
- Downloaded backbone cache revision: `timm/resnet18.tv_in1k`, snapshot `bbd144b3e5565108aad885f145491d11bc6ce807`; metadata also records the torchvision source URL and timm 1.0.31. The full learned checkpoint hash is recorded with the model.
- Checkpoint model version: `98a85da202d7df09`; threshold version: `f66e03aa01039932`. Threshold was selected solely from 12 validation scores: raw distance **1.8165923953056335**, uncertainty margin **0.041223672866821284**. This threshold is **not suitable for real camera deployment**.
- Uncompressed OpenVINO export succeeded. Input `[1,3,128,128]`, `pred_score` `[1]`, `anomaly_map` `[1,1,128,128]`. XML/BIN/JSON exist in `artifacts/models/patchcore/`.
- Separate-process `predict` succeeded on a held-out synthetic defective image; it returned `DEFECTIVE`, raw score approximately **2.85713**, and saved a map/overlay. It did not depend on a training-process Python object.
- Executed PyTorch/OpenVINO parity on the 12 **validation** images: maximum absolute score difference **1.6093254089355469e-06**, zero decision mismatches (tolerance 0.001).
- Independent synthetic test report was generated with TN=6, FP=0, FN=0, TP=6 and no uncertainty abstentions. These deliberately simple drawings are a **software smoke test, not real defect accuracy or a challenge-performance claim**. No real-data metrics exist.
- Scripted mock demo: 3/3 cycles, including good, defective and one reinspection, no errors/timeouts.
- Actual OpenVINO + MockRobot directory demos: 6/6 good cycles and 6/6 defective cycles, no errors/timeouts. Physical task success remains unverified.
- CPU benchmark: 5 warmups, 30 measured calls, one fixed validation image; mean **5.172 ms**, p50 **5.238 ms**, p95 **5.661 ms**. Measures `infer()` only, excluding decode/resize/compile/overlays. The optional simulation was running during this measurement; this is a local observation under that workload, not an isolated hardware benchmark or Intel prediction.
- Optional MuJoCo: isolated plugin 0.5.0 / runtime 0.4.0 / MuJoCo 4.0.1 installed, SO-101 scene started, browser arm/cube/target verified, wrist and overview rendering confirmed by health response. Stopped afterward. It is not integrated with Brickcell's controller.

## Test and tooling results

Final default offline run: **44 passed, 3 skipped, 0 failed** in 1.79s. The three skips are intentional model-asset tests gated by `BRICKCELL_MODEL_CONFIG`.

Final opt-in offline full run: **47 passed, 0 skipped, 0 failed** in 4.73s. Two upstream deprecation warnings (`timm.models.layers` and Anomalib's DINOv2 components) remain; neither affects PatchCore tests. Earlier suites also passed as tests were added; there were no unresolved pytest failures.

Ruff check passed and all 32 Python files passed formatting. Installed API signatures/source hashes were captured in `artifacts/api_signatures.json`. JUnit outputs are `artifacts/tests-software.xml` and `artifacts/tests.xml`.

Coverage includes directory importing, corrupt/missing images, duplicate hashes/IDs/paths, normal-only fitting, object/session overlap, deterministic grouped splits, changed-image provenance, RGB/channel/range handling, quality gates, threshold calibration/uncertainty/missing threshold, malformed input/result data, robot state transitions, good/defective sorting, retry budgets, action and inspector timeouts, stale results, failed grasp/release, safe stop after logging errors, full CLI mock integration, OpenVINO inference and numerical parity.

## Commands actually executed

Commands were run from the repository root. Agent executions used `UV_CACHE_DIR=/tmp/brickcell-uv-cache`; native OpenVINO commands used approved execution outside the restrictive agent sandbox.

```bash
uname -a
python3 -c 'import platform,sys; print(platform.machine()); print(sys.version)'
uv --version
uv python list --only-installed
sysctl -n machdep.cpu.brand_string
uv sync --python 3.12
uv sync --locked --offline
uv run python -m brickcell doctor
uv run python -m brickcell synthetic
uv run python -m brickcell validate-data --manifest data/synthetic/manifest.json
uv run python -m brickcell demo --mock
uv run python -m brickcell train --config configs/patchcore.yaml
uv run python -m brickcell export --config configs/patchcore.yaml
uv run python -m brickcell parity --config configs/patchcore.yaml
uv run python -m brickcell predict --model artifacts/models/patchcore/model.xml --image data/synthetic/test/defective/synthetic-test-defective-000.png
uv run python -m brickcell evaluate --config configs/patchcore.yaml
uv run python -m brickcell benchmark --model artifacts/models/patchcore/model.xml --image data/synthetic/val/good/synthetic-val-good-000.png --iterations 30 --warmup 5
uv run python -m brickcell demo --model artifacts/models/patchcore/model.xml --images data/synthetic/test/good --output artifacts/demo-good
uv run python -m brickcell demo --model artifacts/models/patchcore/model.xml --images data/synthetic/test/defective --output artifacts/demo-defective
uv run pytest -m 'not model' -q
uv run --offline pytest -q --junitxml=artifacts/tests-software.xml
BRICKCELL_MODEL_CONFIG=configs/patchcore.yaml uv run --offline pytest -q --junitxml=artifacts/tests.xml
uv run ruff check src tests scripts --fix
uv run ruff format src tests scripts
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/inspect_apis.py
uv venv artifacts/mujoco-venv --python 3.12
uv pip install --python artifacts/mujoco-venv/bin/python physicalai-mujoco-plugin
artifacts/mujoco-venv/bin/physicalai-mujoco start --help
MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" PHYSICALAI_MUJOCO_CACHE_DIR="$PWD/.cache/physicalai_mujoco" artifacts/mujoco-venv/bin/physicalai-mujoco start --profile so101 --name brickcell-so101-smoke --seed 42 --status-json --http-port 18080 --viser-port 19090 --exit-with-parent
curl --fail --max-time 5 -sS http://127.0.0.1:18080/health -o artifacts/simulation/health.json
uv pip freeze --python artifacts/mujoco-venv/bin/python
```

Grouped import/split CLI functions and failure cases were executed by pytest using temporary fixtures. The native device probe, source/signature inspections and Git hygiene checks were also executed. The simulation startup command was attempted twice and then stopped with Ctrl+C after verification.

## Issues encountered and resolved

1. **Restricted sandbox OpenVINO crash:** a native `Xbyak_aarch64::Error: internal error` terminated the initial device probe. Approved native local execution succeeded. Doctor uses child processes so it can report this cleanly. This is not evidence that the M4 lacks OpenVINO CPU support.
2. **ONNX opset mismatch:** the first successful export inherited Anomalib's default opset 14 and logged a failed downgrade from exporter opset 18, retaining the newer graph. The wrapper now explicitly requests opset 18; re-export and numerical parity succeeded without that downgrade attempt. The initial message is retained in the append-only export log.
3. **MuJoCo first-start deadline:** initial scene startup exceeded the external plugin's 10s owner deadline. The second attempt with cached assets/imports succeeded. No plugin source patches were applied.
4. **Agent cache permissions:** the global uv cache was not writable in the sandbox. A task-specific `/tmp` cache and project-local library caches resolved this without changing `HOME`.

## Created files and remaining work

Created `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `README.md`, `AGENTS.md`; model/inspection/controller configs and an isolated MuJoCo requirements snapshot; the `src/brickcell` package (data, detector, inspection, sources, controller, metrics, CLI/doctor/utilities); six test files; `scripts/inspect_apis.py`; and architecture, collection, deployment, simulation and validation documentation. Existing `ssh.md` is unchanged and was not used for connections during this task.

Generated data, models, reports, logs, caches and environments are local and Git-ignored. No commits/pushes were made. No API keys/private keys were added.

Still unverified: real brick/camera accuracy; real-data threshold and quality settings; calibrated multi-view capture; physical grasp/release/placement; SO-101 hardware adapter, trajectories, emergency stop and joint-unit checks; Intel CPU/GPU/NPU deployment/precision/throughput; Studio policy training/export; Modal jobs; and a controller adapter for the optional MuJoCo scene. Linux compatibility is resolved but not runtime-tested. See `INTEL_DEPLOYMENT.md` for the staged hardware plan.

## Follow-up live demonstration

On request, reran `demo --mock` (3 completed cycles, 1 reinspection, 0 errors) and the OpenVINO defective-image demo (6 completed cycles, 0 errors). Restarted the SO-101 simulation, selected **Conveyor Sort** in the viewer and enabled **Autopilot → Drive the arm**. The local health endpoint confirmed `scene=conveyor_sort`, `autopilot.mode=drive`, and an active `carry` phase targeting `item_hex_red_cracked`, at approximately 50 Hz with both cameras around 30 fps. The viewer was opened and left running at `http://127.0.0.1:19090` for the user. This motion is the plugin's scripted demonstrator, not Brickcell/OpenVINO-controlled motion. Snapshot: `artifacts/simulation/live-health.json`. Use the viewer's Shutdown control to stop it.

## Camera-driven two-bowl implementation — 14 November 2026

This supersedes the earlier note that MuJoCo was not integrated. The new scene and bridge run native ARM64 on the same M4 Max. `doctor` was rerun: Python 3.12.11, Anomalib 2.7.0, torch 2.10.0, torchvision 0.25.0, OpenVINO 2026.4.1; OpenVINO devices **CPU** only, MPS available but unused. Report: `artifacts/two-bowl/doctor.json`.

Created `src/brickcell/simulation/{world,generate,server,bridge,protocol,run}.py` and package initializer; `configs/patchcore-sim.yaml`, `configs/inspection-sim.yaml`, and `tests/test_simulation.py`. Extended `tests/test_model.py` and pytest markers in `pyproject.toml`; updated README, architecture and simulation documentation. No detector or supervisor implementation changes were required. Dependencies remain locked; `uv sync --locked` succeeded after the project metadata update.

Executed results:

- Rendered 48 camera fixtures using MuJoCo contact grasps: 24 train-good, 12 validation, 12 test. Dataset validation: no missing/corrupt images, duplicates or partition overlap. Synthetic cloned geometry, not real data.
- PatchCore training succeeded; model version `8e92730963cde567`. Only good images fitted the memory bank. Validation-only threshold **1.3344924449920654**, uncertainty margin **0.03300517779588699**; threshold version `8d65f6c4bb0254c4`.
- Uncompressed OpenVINO export and independent-process CPU inference succeeded. Input `[1,3,128,128]`, score `[1]`, anomaly map `[1,1,128,128]`.
- PyTorch/OpenVINO parity on 12 calibration images: **maximum absolute difference 0.000007241964340209961**, zero decision mismatches.
- Frozen-threshold held-out synthetic test: TN=6, TP=6, FP=0, FN=0, no uncertainty abstentions. This is a software consistency check, not a real defect-performance result. Mean CPU inference 9.164 ms, p50 5.588 ms; includes first-call warmup, only 12 samples, no p95 claim.
- First live camera/physics run: **4/4 completed**, GOOD=2, DEFECTIVE=2, zero reinspection/timeouts/errors. Scores by opaque part ID: 0.513532, 2.089751, 0.513414, 2.056424. The arm used actual contact grasps, and each release was checked inside its commanded bowl. Mean cycle 17.538 s (deliberately paced simulation). Report: `artifacts/two-bowl/runs/20261113-235951-952892/summary.json`.
- Final software suite: **56 passed, 5 skipped**, zero failures, 1.97 s. Skips are four explicit model-asset tests and one live-simulator test; JUnit: `artifacts/two-bowl/tests-software.xml`.
- Opt-in model and live-simulation suite: **5 passed, 56 deselected**, zero failures, 84.03 s. Includes a full second four-object contact-physics run and a causality test that changes only image pixels while retaining an identical object ID; the selected bowl changes. Two upstream deprecation warnings remain. Report: `artifacts/two-bowl/tests/latest.json`.
- Ruff check and formatting passed for all 40 Python files. No physical hardware success or Intel acceleration was claimed.

Commands executed for this addition (from the repository root; native OpenVINO/rendering commands used approved local execution):

```bash
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.generate --output data/two-bowl
uv run python -m brickcell train --config configs/patchcore-sim.yaml
uv run python -m brickcell export --config configs/patchcore-sim.yaml
uv run python -m brickcell parity --config configs/patchcore-sim.yaml
uv run python -m brickcell evaluate --config configs/patchcore-sim.yaml
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.server
uv run python -m brickcell.simulation.run --reset
uv sync --locked
uv run pytest --junitxml=artifacts/two-bowl/tests-software.xml
BRICKCELL_MODEL_CONFIG=configs/patchcore-sim.yaml BRICKCELL_QUALITY_CONFIG=configs/inspection-sim.yaml BRICKCELL_SIMULATION_URL=http://127.0.0.1:18081 uv run pytest -m 'model or simulation'
uv run python -m brickcell doctor --model artifacts/models/patchcore-sim/model.xml --output artifacts/two-bowl/doctor.json
uv run python -m brickcell validate-data --manifest data/two-bowl/manifest.json --output artifacts/two-bowl/data-validation.json
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

During development, an unreachable initial IK branch was fixed by initializing a reachable ready pose; the grasp yaw was changed to expose the inspection face; separate drop offsets avoid stacking; settling is checked before completion. A browser replay launch initially resolved the venv Python symlink into the bare interpreter, producing `ModuleNotFoundError: yaml`; preserving the venv executable path fixed that issue. Failed development attempts are not counted as successful runs.

Limits: one visible synthetic defect face, three named virtual cameras but only inspection used for inference, scripted IK and known pickup poses, no learned VLA or real-camera picking. The model must be retrained/calibrated on real station images. Real arm calibration, hardware driver/policy integration, camera alignment, safe motion, emergency stop and Intel device benchmarking remain unverified.

The final browser replay button was exercised after fixing its launcher: 4/4 completed, GOOD=2, DEFECTIVE=2, zero errors/timeouts/reinspections. Report: `artifacts/two-bowl/runs/20261114-000814-770102/summary.json`. The connected viewer visibly reported completion and was left open at http://127.0.0.1:19091/. Screenshot: `artifacts/two-bowl/completed-view.jpg`.

## Five-face contract, ACT preparation and sharing — 14 November 2026

The user selected four sides plus the studded top, excluding the bottom; marker marks/surface contamination first. Their photo shows square and long/narrow colored bricks in several orientations. They estimated approximately 20–40 mm, not measured dimensions. `configs/objects.yaml` preserves that uncertainty.

Implemented and tested `CoverageLedger`: all five face results are needed for GOOD, invalid/missing/uncalibrated results require reinspection, errors stop sorting, identities and duplicate frames are checked, and detected defects remain latched. **This ledger is not yet connected to a five-face presentation/re-grasp sequence. The live demo remains single-view.**

Added an isolated official ACT lab, demonstration schema/recorder, teacher-free simulation rollout worker, training/evaluation scripts, object/coverage configs, Makefile and team handoff documentation. `policy/uv.lock` pins the training environment separately from `uv.lock` and the simulator snapshot. No paid/cloud deployment or hardware connection was made.

Executed evidence:

- `record_policy` produced six successful synthetic scripted-teacher episodes with 2,250 synchronized frames: train=4 episodes, val=1, test=1. Both cameras, six observed joints, six pre-step commanded joint targets and timestamps are archived with hashes. This is not human teleoperation or a LeRobot/Studio UI dataset.
- Official Physical AI Studio ACT CPU training completed 50 steps in 12.52 seconds; first/last training losses 7.72383/0.61693, seeded validation-batch action MAE 0.07099 radians, exact checkpoint reload prediction match. This measures a learning pipeline, not manipulation success. Metadata/report: `artifacts/policy/act-smoke/report.json`.
- Strict teacher-free rollouts: 0/2 success, rejected out-of-limit predictions at step 0. Diagnostic rollouts with explicitly enabled simulator-limit saturation: 0/2 success after 600 steps each, 600 clipped action values per episode. Reports: `rollouts.json` and `rollouts-bounded.json` alongside the checkpoint. The scripted controller was never used for rollout actions. This smoke policy is **not deployable**.
- The initial ACT constructor rejected list-shaped feature dimensions; the wrapper was corrected to the installed API's tuple requirement. macOS PyAV/OpenCV imports emit duplicate AVFoundation-class warnings; completed training/rollouts were not blocked by them.
- Final offline software tests: **69 passed, 5 skipped, zero failures**, 2.03 seconds; `artifacts/tests-policy-prep.xml`. The skips remain four explicit model tests plus the live-simulator test, whose earlier successful executions are recorded above. No full model/live rerun was needed for documentation, independent coverage/data logic, or the default no-op recording hook.
- Ruff check and format passed for 49 Python files including the separate policy scripts.
- Wheel and source archive built. Source archive inspected: 74 entries, no `ssh.md`, datasets, artifacts or virtual environments. Wheel contains the data tools and coverage ledger. Installed the wheel without dependencies in a fresh temporary environment and ran `python -I -m brickcell --help`; this tests package/CLI discovery, not a fresh installation of every dependency.
- Packaging initially included the existing unrelated `ssh.md`; the sdist now has an explicit project-file allowlist. Existing repository history/file content was not altered.

Principal executed commands:

```bash
uv sync --project policy --python 3.12
PYTHONPATH=src MENAGERIE_CACHE_DIR="$PWD/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.record_policy
PYTHONPATH=src TORCH_HOME="$PWD/.cache/torch" policy/.venv/bin/python policy/train_act.py
PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py
PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py --clip-to-sim-limits --output artifacts/policy/act-smoke/rollouts-bounded.json
uv sync --locked
uv run pytest --junitxml=artifacts/tests-policy-prep.xml
uv run ruff check src tests scripts policy
uv run ruff format --check src tests scripts policy
uv build --out-dir artifacts/dist
uv venv artifacts/package-check --python 3.12
uv pip install --no-deps --python artifacts/package-check/bin/python artifacts/dist/brickcell-0.1.0-py3-none-any.whl
artifacts/package-check/bin/python -I -m brickcell --help
git switch -c codex/physical-ai-prototype
```

No hosted deployment is required to share the feature branch. The next useful work is reviewed data, meaningful learned-policy rollouts, a verified five-face presentation plan, and real station data/adapter integration. See `POLICY_LAB.md` and `TEAM_HANDOFF.md`.

## SO-101 manual geometric alignment on Intel — 14 November 2026

Executed on the team's Ubuntu Intel computer through authorized SSH access,
using Python 3.12.16. Raw acquisition used the existing LeRobot/servo-SDK uv
environment; independent kinematic checks used installed Frax 0.2.0 with JAX
64-bit precision. No root environment or Mac CPU dependency pins were changed.

Both arms' five direction signs were recorded as +1, and increasing gripper
values were observed to open the jaw. Before/after raw snapshots, rejected
isolation attempts and accepted results are retained under
`calibration/measurements/manual-direction-session/`. The leader elbow sign was
explicitly accepted by the user despite shoulder co-movement; it is not labelled
as a strict-isolation pass.

The user manually matched each arm by eye to the fixed CAD target
`[0, 93.1, -89.5, 1.5, 90]` degrees. A read-only capture derived the per-arm
encoder-zero offsets, preserving both original LeRobot calibration files and
their hashes. Both alignment flags are confirmed for this **manual visual
method**, whose physical uncertainty is unquantified. See
[`calibration/ALIGNMENT_REPORT.md`](../calibration/ALIGNMENT_REPORT.md).

Generated `models/so101-{leader,follower}-calibrated.urdf` contain five bounded
arm coordinates and fixed CAD hub/gripper coordinates. They are massless
kinematics-only models; effort/velocity, TCP and camera mounting transforms
were not measured. They are not contact-dynamics models or validated motor
command limits.

Executed checks:

- Both actual aligned models: Frax FK/Jacobian agreement, central-difference
  Jacobian checks and per-arm position-bound checks at 10 configurations each.
  Numerical errors were below 2e-8; this is software consistency, not measured
  physical pose accuracy. Machine-readable reports are in `models/`.
- Calibration/kinematics unit checks: **8 passed, 1 skipped** in each of the two installed interpreters.
  Each optional backend is tested where installed; the other backend is skipped
  in that interpreter. This includes rejection of stale calibration and CAD
  model bindings, angle-unit conversions, and Jacobians with TCP offsets.
- Latest upstream core software tests in an isolated uv environment on Intel:
  **69 passed, 5 skipped**, 2.28 seconds. Four skips require trained model assets;
  one requires a live simulator. Neither was started for this task.
- `ruff check src tests scripts` and `ruff format --check src tests scripts`
  passed (47 files). These files were the incoming prototype revision, not
  modifications to the detector/controller during calibration.

The core checks used a temporary source archive and minimal test dependencies
to preserve the working Intel XPU environment: pytest 9.2.0, Ruff 0.17.5,
NumPy 2.5.4, Pillow 13.0.0, PyYAML 6.0.3, OpenCV headless 5.0.0.93,
scikit-learn 1.9.1 and SciPy 1.18.1. Initial collection omitted scikit-learn;
adding that declared dependency resolved collection and the suite passed.
This was not a reinstallation or validation of the complete locked anomaly
model stack on Intel.

Reproducible arm checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tools -p 'test_so101_*.py' -v
PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 ~/frax/.venv/bin/python -m unittest discover -s tools -p 'test_so101_*.py' -v
JAX_ENABLE_X64=1 ~/frax/.venv/bin/python -B tools/so101_validate_alignment.py
```

All physical movements were made by people, with torque off. Acquisition sent
only READ instructions; no homing, limit, torque or goal-position registers
were written and no robot services were started/stopped. No automatic physical
pose command, grasp, inspection or sorting success was tested. A subsequent
supervised physical pose/TCP check is still needed to measure the alignment error.

## Offline data/training workbench — 2026-11-14

Built on `codex/data-training-workbench`, initially from `6408aad` and fast-forwarded
to the team's non-overlapping alignment commit `3526b29` before publication.
No changes to arm-api, calibration files, or hardware-control scripts are part of
this workbench diff. No hardware connection or cloud training was performed.

Executed on the existing native M4 / Python 3.12 environment:

- Core pytest: **103 passed, 5 skipped** (34 workbench tests). Default skips are four
  model tests and one live-simulator test. Ruff lint/format checks passed.
- Separately ran the existing live MuJoCo/OpenVINO sorting integration test against
  an isolated local simulator on ports 18083/19093: **1 passed, 105 deselected,
  71.32 seconds** at that invocation. This verifies simulated sorting after UI
  changes, not physical success. The isolated test server was stopped afterward.
- Generated a new offline fixture, imported its 36 inspection images through CSV,
  preserved family/color metadata, and reviewed its three synchronized 30-frame
  demonstration-format episodes. Strict readiness passed for these software
  fixtures. They do not represent a learned grasp or realistic defect benchmark.
- Reviewed the existing six MuJoCo teacher episodes (2,250 frames) and the 48-image
  two-bowl camera dataset. Inspection remained smoke-trainable but lacked named
  five-face metadata. All six policy episodes failed the conservative joint-limit
  readiness gate. No archived data or limits were altered to conceal findings.
- Compared directly with the composed simulator model: largest wrist-flex observed
  excess per episode ranges from **0.000158 to 0.003070 rad**; commanded excess is
  about **0.00000274 rad**. These small boundary/precision/soft-constraint effects
  are not evidence of collision or an explanation for the earlier failed ACT
  rollouts. The report gives the exact frame indices, bounds, maximum excess and
  explicit numerical tolerance (1e-6 rad) for review.
- Used the emitted inspection config to train PatchCore on the imported synthetic
  fixture successfully. Used the emitted policy command with `--steps 2` to run
  an ACT CPU optimizer/checkpoint-reload smoke on the format fixture successfully
  (reload maximum difference 0). Neither measurement is a real-task success claim.
  The known PyAV/OpenCV duplicate AVFoundation warnings appeared in the isolated
  policy environment; the command completed.
- Browser verification: next-frame stepping advanced both camera frames and joint
  values; Play/Pause advanced to frame 123; selecting the validation episode reset
  playback to its own frame zero. Image label/split filters returned six defective
  validation images; selecting the absent top face returned zero. Final responsive
  layout and compact metadata cards were inspected. The simulator visibly showed
  synthetic/scripted mode and one fixed view with **0/5 confirmed named faces**.
- Wheel and source archive built successfully. The first sandbox build could not
  resolve PyPI; approved native execution completed the build. No dependency or
  lockfile changes were necessary.
- Extracted the source archive into a clean temporary directory, installed with
  `uv sync --locked` into a new environment, and successfully ran fixture generation,
  CSV import, and both strict review commands. The archive includes workbench
  modules, configs and handoff docs; datasets, artifacts and unrelated `ssh.md`
  are excluded. The wheel includes all eight workbench modules.

Representative executed commands (from repo root; output directories must be new):

```bash
uv run pytest
uv run ruff check src tests scripts policy
uv run ruff format --check src tests scripts policy
uv run python -m brickcell.workbench demo --output data/workbench-example
uv run python -m brickcell.workbench import-inspection --csv data/workbench-example/inspection/capture.csv --output data/workbench-example/inspection/imported.json
uv run python -m brickcell.workbench inspection --manifest data/workbench-example/inspection/imported.json --output artifacts/workbench/example-inspection --strict
uv run python -m brickcell.workbench policy --manifest data/workbench-example/policy/manifest.json --limits configs/workbench-sim-limits.json --output artifacts/workbench/example-policy --strict
uv run python -m brickcell train --config artifacts/workbench/example-inspection/patchcore-review.yaml
PYTHONPATH=src TORCH_HOME="$PWD/.cache/torch" policy/.venv/bin/python policy/train_act.py --manifest data/workbench-example/policy/manifest.json --output artifacts/workbench/example-policy/training-run --steps 2
BRICKCELL_MODEL_CONFIG=configs/patchcore-sim.yaml BRICKCELL_QUALITY_CONFIG=configs/inspection-sim.yaml BRICKCELL_SIMULATION_URL=http://127.0.0.1:18083 uv run pytest -m simulation -q
uv build --out-dir artifacts/workbench-dist
```

Real recording-format conversion, sensor synchronization, physical coordinate and
limit profiles, camera visibility, five-face robot presentations and reliable
learned motion still need station validation. Static reports suit small pilot
batches; they embed every preview frame and are not a large-dataset streaming UI.
See `WORKBENCH.md` and `STATION_COLLECTION.md` for the reproducible workflow and
operator handoff.

## Arm camera input in the RGB/depth viewer — 14 November 2026

Added a configured `wrist` RGB-only input beside top and plate, with an arm/wrist
card, native-resolution crop, separate real RGB model contract, calibrated
threshold decision and anomaly overlay. RGB-only inspection does not fabricate
or require depth. D415 inspection still requires both RGB and depth channels.
The existing remote Sonix helper remains the sole camera owner.

Executed locally with Python 3.12.16 and the locked environment:

- `uv sync --locked` completed; no dependency changes.
- `uv run pytest`: 154 passed, 5 skipped. Four trained-model tests require
  `BRICKCELL_MODEL_CONFIG`; one live simulator test requires its running service.
- `uv run ruff check src tests scripts`: passed.
- `uv run ruff format --check src tests scripts`: passed (88 files).

Tests use artificial frames, injected detectors and HTTP responses/handlers.
They cover RGB-only good/defective/uncalibrated decisions, missing models/crops,
wrong camera identity, stale frames, rejecting depth models for wrist, unchanged
RGB pixels, 1920×1080 crop handling and missing acquisition metadata blocking
inference while retaining preview. Existing depth preparation/decoder tests pass.

Not executed: SSH deployment, camera access, remote helper changes, live browser
review, wrist model training/calibration, real OpenVINO inference or physical
sorting. The default wrist model mapping is empty. The legacy `/frame.jpg` source
needs acquisition-time `X-Frame-Time`, `X-Frame-Id` and `X-Camera-Serial` headers
before trusted wrist inference is enabled; arrival time is not capture time.
Without these headers it is preview-only/REINSPECT. No real defect accuracy or
cross-camera identity/coverage is established by these software checks.

Push verification: checked the isolated arm-camera commit contents against HEAD
without other workspace changes: **140 passed, 5 optional tests skipped**;
Ruff lint passed and formatting passed for all 81 committed Python files.
The 154-pass result above includes other pre-existing local work.

## Follower API reference command — 14 November 2026

Executed over SSH on Intel using the existing real-hardware armd environment.
Started the previously stopped API in its dedicated tmux session; both arms
reported online and follower servos reported no errors before motion.
Commanded the saved upright reference raw ticks through `POST
/arms/follower/positions`: pan 1970, lift 2396, elbow 1984, wrist flex 1759,
wrist roll 3220, speed 100 ticks/s, acceleration 10, tolerance 10 ticks,
timeout 20 seconds. No targets were clamped; gripper was omitted.

The API returned `reached: false`. Final ticks before stop were respectively
1919, 2395, 2024, 1779, 3222 (errors -51, -1, +40, +20, +2 ticks).
Sent `/stop` with `mode: hold`; no automatic retry or torque-off was sent.
The hold endpoint also enabled gripper torque at its current 1462 ticks.
Stopped the task-started API with its configured `on_shutdown: hold` behavior.
Human visual comparison remains pending; this attempt does not establish
finished calibration or measured physical pose accuracy. No software was changed;
pytest, model tests and Ruff were not run for this hardware-only operation.

### Supervised elbow correction trial

The user reported the follower visually matched, with joint 3 approximately
2 degrees low, and requested alternative angle commands. Tried one upward
elbow correction from fresh position 2027 to 2004 ticks (approximately -2.02
degrees; increasing ticks lowers the forearm). API speed 80 ticks/s,
acceleration 10, tolerance 5 ticks, timeout 5 seconds. Target was not clamped,
but `reached` was false and the final encoder remained 2027 ticks. Sent hold
and ended the run without another motion attempt. Stopped the task-started
armd session, preserving torque under `on_shutdown: hold`. This trial did not
produce a new achieved pose for visual ranking; the cause of the lack of
movement is undiagnosed. No calibration values or software were changed;
software/model tests and Ruff were not run for this hardware-only trial.

Read-only follow-up: opened the unowned follower port with the existing armd
SDK and issued only register reads, then closed it. All five arm servos read
P=16, D=32, I=0, both directional deadbands=1 tick, minimum startup force=16,
maximum torque=1000, runtime torque limit=1000, position mode=0 and torque
enabled=1. Thus a configured 23-tick deadband does not explain the failed
small move. Weak correction under friction/load remains a hypothesis, not a
confirmed cause. The earlier stop overwrote the goal register, so these reads
cannot establish whether the original 2004-tick goal reached the servo.
The API wait implementation returns false when encoder error stays outside
tolerance until its deadline; the recorded HTTP response was 200, not a
network timeout. No new motion or setting writes were made during diagnosis.

### Elbow goal-readback diagnostic

At the user's request, repeated a bounded small upward command through the
real API, with fresh feedback before and after. Initial elbow position was
2029 ticks; target 2006 ticks, speed 80 ticks/s, acceleration 10, unclamped,
`wait: false`. Eleven fresh samples over 5.051 seconds showed goal=2006 and
position=2029 throughout, speed=0, torque enabled, position mode, no errors,
12.2 V and 31 degrees C. By 0.510 seconds reported load changed from -2.0%
to -10.4% and current from 0 to 39 mA, then settled at 32.5 mA.
The goal readback confirms successful goal delivery for this trial; the
unchanged encoder explains why a wait with 5-tick tolerance would time out.
Corrective effort without displacement is consistent with friction/load and
controller tuning, but does not identify a unique physical cause. Sent hold
at 2029 ticks and stopped the task-started armd session with hold-on-shutdown.
No retries, calibration changes or gain/torque-setting changes were made.
No software changed; pytest, model tests and Ruff were not run.

### Supervised elbow proportional-gain trial

The user authorized the proposed modest P-gain trial. With armd stopped and
the follower port unowned, backed up elbow registers 3–39 and all joint
feedback using the existing SDK. The initial pre-check stopped without any
writes because EEPROM lock register 55 was already 0 (unlocked), rather than
the assumed 1. Evidence: `calibration/measurements/elbow-p20-trial-20261114-01.json`.

For the actual trial, set the lock to 1, set a present-position hold goal,
changed only elbow P from 16 to 20 and verified readback. I=0, D=32 and
calibration offsets were unchanged. Commanded 2033 to 2010 ticks at speed
80 ticks/s and acceleration 10. All 51 fresh samples across 5.047 seconds
read position 2033 with goal 2010; the 5-tick target tolerance was not met.
Reported speed remained zero, no faults were reported, voltage was 12.2 V,
temperature 31 degrees C, current peaked at 58.5 mA, and final trial load
was -12.6%. Increased P produced more effort but no measured displacement;
this does not distinguish friction/load from other servo-control causes.

Issued a hold at 2033, restored P=16, restored original lock=0, and verified
P/D/I, deadband, mode and lock readbacks matched the backup. Final feedback
showed position=goal=2033, speed=0, moving=false, torque enabled, no faults.
No EEPROM unlock was issued; persistence across a power cycle was not tested.
Evidence: `calibration/measurements/elbow-p20-trial-20261114-02.json`.
Removed temporary remote scripts/logs after copying both evidence records.
No software changed; pytest, model tests and Ruff were not run.

### Supervised elbow integral-gain trial

The user authorized a small I-term test. With armd stopped and the follower
port unowned, backed up register bytes 3–39 and all joint feedback. Verified
P=16, D=32, I=0, position mode and original lock=0. Set lock=1 for the trial,
issued a present-position hold, set I=1 and verified readback, then commanded
elbow position 2023 to 2000 ticks at speed 80 ticks/s, acceleration 10.
Recorded 51 fresh samples over approximately 5.05 seconds with bounded
position/current/load monitoring. First sample within 5 ticks of target was
at 4.44 seconds (position 2002). Final trial position was 2002, a 21-tick
movement (approximately 1.85 degrees) and 2-tick residual (0.18 degrees).
No fault or monitoring-limit violation occurred. This short test supports
integral action as helpful in this pose, but does not validate gains across
loads/poses, sustained holding, or physical calibration accuracy.

Issued hold at 2002 and restored I=0 and original lock=0. Readback verified
P=16, D=32, I=0 and unchanged deadbands/mode/lock. Final feedback showed
position=goal=2002, speed=0, moving=false, torque enabled and no faults.
Calibration offsets were unchanged. Evidence:
`calibration/measurements/elbow-i1-trial-20261114-01.json`.
Removed the temporary remote trial script/log/evidence copy after preserving
the local evidence. No software changed; pytest, model tests and Ruff were
not run for this hardware operation.

### Persistent follower I=2 and supervised pose adjustment

The user confirmed the I=1 corrected pose looked better, then explicitly
requested permanent I=2 on all joints, a slightly higher joint-3 target,
and joint 1 approximately 8 degrees more clockwise. Applied this to all six
follower servos, including gripper; leader was untouched. Interpreted the
elbow adjustment as another 2 degrees upward from the fresh held position.

With armd stopped and the follower port unowned, backed up register bytes
3–39 and state for every servo. All six originally read P=16, D=32, I=0,
lock=0. Set present-position hold goals, then wrote I=2 with EEPROM unlocked
and relocked each servo (lock=1). Verified every register byte 3–39 matched
its backup except the intended I register. Monitored the new gains holding
for two seconds before issuing a sync position command to joints 1 and 3.

Motion start was pan=1919, elbow=2003 ticks. Targets were pan=2010 (+91
ticks, approximately +8 degrees clockwise viewed from above) and elbow=1980
(-23 ticks, approximately 2 degrees upward). Speed=80 ticks/s, acceleration=10.
After five seconds, measured pan=2011 and elbow=1983, within the 5-tick
tolerance. Measured movement was +92 ticks (8.09 degrees clockwise) and
-20 ticks (1.76 degrees upward). Other four joints stayed at their hold goals.
Monitoring of all six joints reported no fault or limit violation.

Issued final holds at achieved positions. Final feedback showed every joint
stationary with torque enabled and no faults. Readback verified P=16, D=32,
I=2 and lock=1 on all six. Gain writes were made with EEPROM unlocked for
persistence; power-cycle persistence has not been experimentally tested.
Calibration offsets, mechanical limits and leader settings were unchanged.
This short supervised test does not validate gains for all poses/loads or
long-duration stability. Evidence and original settings are retained in
`calibration/measurements/follower-i2-persistent-20261114-01.json`.
Temporary remote scripts/logs were removed after preserving evidence.
No software changed; pytest, model tests and Ruff were not run.

## SO-101 sourced inertials and trajectory identification — 14 November 2026

Executed on Intel in the separate `dynamics/` Python 3.12.16 environment,
with `uv sync --locked`: MuJoCo 4.0.0, Frax 0.2.0, JAX/JAXlib 0.12.0.1,
NumPy 2.5.4, SciPy 1.18.1 and trimesh 4.12.2. Full manifests are retained in
`artifacts/dynamics/{physics,armd}-packages.txt`. Core anomalib pin and the
existing Intel core/LeRobot environments were preserved.

### Real acquisition

The final user-requested trajectory was 40 degrees peak-to-peak (±20 degrees),
one of five arm joints at a time. Gripper opening was fixed. Used the existing
localhost armd API and its configured nonzero 300 ticks/s default, acceleration
10. Firmware safety registers and armd fault/motion/position checks were not
disabled. The task-specific API launcher selected its existing individually
addressed read fallback without changing deployed armd source/configuration.

At the user's explicit direction, extra recorder-level speed, current, load,
voltage and temperature cutoffs were removed. Recorder position envelopes,
fresh/available position data, actual goal readbacks, finite duration and
return-to-center checks remain. Earlier aborted sessions are retained on
Intel as incomplete records and are rejected by fitting. Stops included a
conservative thermal gate, transient reported speed/temperature readings,
266.5 mA current and a 10.8 V reading without a servo fault. Their exact
electrical/telemetry origin was not established. With the arm explicitly
supported by the user, torque was turned off once for cooldown and verified;
it was re-enabled at present-position goals when the user requested resumption.

Completed fitting capture: `fit-session-10.json`, **1,278 fresh samples over
64.985 s**, 12 s cycles. Actual encoder spans, joints 1–5: **38.67, 39.38,
38.85, 38.41, 37.62 degrees**. Separate validation capture:
`val-session-01.json`, **1,085 samples over 54.956 s**, 10 s cycles; spans
**38.76, 37.18, 38.32, 37.27, 38.76 degrees**. Both completed and held.
PID readback remained P=16/D=32/I=2 with the expected homing offsets; no
gain or calibration changes were made during dynamics collection. Final
feedback showed all six joints stationary, torque enabled and no faults.
The task-started API was then stopped with its hold-on-shutdown behavior.

### Published mass and inertia inputs

Replaced initial heuristic inertials with the official Robot Studio SO101
URDF, pinned to `a758567c3978dfeefe282ede0500085a48fe8f78`, SHA-256
`3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c`.
Source files, license, asset hashes and URLs are cached under ignored
`artifacts/dynamics/upstream/`.

Published physical group masses: base 147 g, shoulder 100.006 g, upper arm
103 g, lower arm 104 g, wrist 79 g, gripper plus jaw 99 g; total **632.006 g
including fixed base**. An earlier conversational total of 732 g was an
addition error; the imported numbers and model use 632.006 g.

COM and full 3x3 inertia tensors were transformed into the existing calibrated
local body frames using axis-constrained surface registration. Jaw inertia
was combined using the parallel-axis theorem; fixed subpart mass is consolidated
into rigid owners. Incoming moving-joint axes are preserved exactly. The four
moving-body outgoing axis-line mismatches were at most **0.332 mm**, axis
angle mismatch at most **0.395 degrees**. Surface-sampling RMS was 2.14–2.82 mm;
this is model registration, not measured physical accuracy. The fixed base's
registration has separate residuals in the input JSON. Kinematic calibration
files and encoder offsets were unchanged.

Menagerie commit `370c43965323c0944b433b202aac5058b5f907a9` provides explicit
simulation assumptions for joint armature 0.028 kg m², damping 0.60,
frictionloss 0.052 and torque cap 2.94 Nm. BAM's pinned Feetech model was also
retrieved, but its 7.4 V/variant conditions were not established as equivalent
to this follower's 12 V hardware. Its constants were not labelled a measured
current-to-torque calibration for this arm.

### Forward dynamics, optimization and visible replay

Native MuJoCo affine P/D feedback is evaluated inside each 1 ms physics step;
the integral/reference update is 100 Hz with clamping and estimated initial
gravity equilibrium. An earlier explicit derivative-torque hold created
artificial wrist oscillation; a regression test now covers this failure.
Masses/COMs/tensors stay frozen at published CAD values while effective SI
actuator gains and training-selected command delay are fitted. SI gains are
not firmware register values and are never written to the real servos.

Final sourced-model fit converged (`ftol`): training RMS error **0.3515 degrees**;
separate validation RMS **0.4360 degrees**, down from **3.0229 degrees** with
the same sourced model's initial gains. Per-joint validation RMS:
**0.275, 0.574, 0.473, 0.363, 0.435 degrees** (consult the machine-readable
report for exact values). Maximum validation error across all joints/frames:
**2.4899 degrees**. Several effective gains reach bounds; numerical Jacobian
rank does not establish unique physical motor constants. The earlier heuristic
inertial fit reached 0.4962 degrees validation RMS and is retained as development
evidence, not the final inertial model.

Final assets: `model-published-verified/`, `published_priors_verified.json`,
`published-fit/fit_report.json` and `phantom-published.html` under
`artifacts/dynamics/` on Intel, with local copies. Input/model hashes, parameter
history and exact package versions are recorded. The HTML autoplays real
encoder FK against simulated motion and exposes initial/intermediate/fitted
stages plus physical-group masses, COMs and full tensors. It was rendered
inline and its geometry/playback smoke check passed. This is recorded encoder
motion, not live camera tracking.

Executed checks on the sourced model: Frax/MuJoCo maximum rigid mass-matrix
difference **2.326e-11**, gravity-vector difference **2.871e-10** over five
configurations; positive mass-matrix eigenvalues. Frax's fixed gravity was
rescaled from 9.81 to 9.80665; MuJoCo armature was removed for the rigid-matrix
comparison. Timestep halving changed joint positions by at most
**7.511e-5 rad**. Numerical freefall error was 0.490 mm after 0.1 s; a sphere
settled at 24.981 mm against its 25 mm radius with one contact.

Optional pipeline tests: **15 passed**; Ruff check and format passed (all pipeline
files). Full core `uv sync --locked`, `uv run --locked pytest` on an isolated
copy on Intel: **69 passed, 5 skipped**; all 47 core files passed
`ruff check src tests scripts` and `ruff format --check src tests scripts`.
Four skips need trained model assets; one needs the live two-bowl simulator.
Neither was started for this task. Temporary core test checkout was removed
after preserving its log.

Limits: published CAD nominals are not per-unit weighings, and real trajectories
do not uniquely identify free masses and uncalibrated actuator torque scale.
No external link/TCP metrology or final frozen test across payloads was done.
The model fixes jaw pose and lacks identified backlash/thermal/electrical
dynamics. **Contact identification is outside the requested scope**; numerical
contact fixtures do not establish real contact/material parameters. No Intel
GPU/NPU benchmark or physical grasp/sort claim follows from this work.

## Autonomous 3D DUPLO localization & 8-corner inspection architecture (Windows local) — 14 November 2026

Implemented locally on Windows (`win32` / `AMD64`) without SSH connections, physical robot commands, real camera capture, or paid cloud execution. Root `pyproject.toml` (`anomalib==2.7.0`) and `uv.lock` were left untouched; the local development environment was provisioned separately under ignored `artifacts/arch-venv/` (`Python 3.12.14`) and snapshotted in `configs/arch-local-requirements.txt`:

- Core versions verified: `mujoco==4.0.1` (`mujoco-core==4.0.1`, `mujoco-studio==4.0.1`, `mujoco-menagerie==2026.9.4`), `frax==0.2.0`, `elastiqp==0.2.1`, `jax==0.12.0.1`, `jaxlib==0.12.0.1`, `babyros==0.1.10`, `eclipse-zenoh==1.10.2`, `openvino==2026.4.1`, `modal==1.6.4`, `qualia-cli==0.3.0rc16`, `qualia-sdk==0.8.0rc16`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `trimesh==5.2.1`, `opencv-python-headless==5.0.0.93`, `pytest==9.2.0`, `ruff==0.17.5`.
- Extracted official LDraw CC-BY-4.0 LEGO DUPLO meshes (`3437` 2x2, `3011` 2x4, `76371` 1x2x2, `2302` 2x3 curved top) into `assets/duplo/`.
- Implemented `src/brickcell/block_inspector/` (`geometry.py`, `ledger.py`, `frax_ik.py`, `perception.py`, `cogman.py`, `twin.py`, `runtime_bus.py`, `armd_bridge.py`, `camera_adapters.py`, `policy_skills.py`, `cli.py`), edge/container manifests (`wendy.json`, `docker-compose.intel-local.yaml`, `services/*`), Qualia task definition (`configs/qualia/inspection_task.json`), Modal cloud orchestration (`cloud/modal_app.py`), and `tests/test_block_inspector.py`. VLM inference is intentionally omitted per task scope while keeping `target_selector_fn` and Qualia manifest hooks ready for camera-world integration.

### Executed checks and simulation evidence

- Full repository pytest suite (`artifacts\arch-venv\Scripts\pytest.exe -q`): **77 passed, 5 skipped, 0 failed** in 46.43s (`69` existing Brickcell software tests + `8` new tests in `tests/test_block_inspector.py`; the `5` skips are the existing opt-in `model` and `two-bowl` live-simulator markers).
- Standalone suites: `arm-api/tests/test_arm.py` (**29 passed** in 1.16s) and `tools/test_so101_*.py` with Frax 0.2.0 `JAX_ENABLE_X64=1` (**8 passed, 1 skipped** in 4.75s).
- Lint & formatting: `ruff check src tests scripts` and `ruff format --check src tests scripts` passed across all 61 Python files.
- Manifest & Frax asset validation (`python -m brickcell.block_inspector validate`): exit code 0; exported `artifacts/block_inspector/frax/{so101_follower_frax.urdf,so101_spheres.yaml}` and validated `wendy.json`, `configs/qualia/inspection_task.json`, and all 4 DUPLO STL assets.
- Synchronized station cameras & verified dynamics (`14 November 2026` update):
  - Loaded measured station camera poses and pinhole intrinsics from `configs/sim-scene.yaml` (`origin/sim/station-twin`: `"Top"` `overhead` D415 `640x480`, low `"Plate"` `side` D415 `640x480` at `[0.3699, -0.4686, 0.2189]`, and `"Gripper"` `wrist` `640x360` mapped onto `Body3` via `registrations["Body3"]`).
  - Wired `artifacts/dynamics/published_priors_verified.json` directly into the MuJoCo 4.0.1 cell (`T_WORLD_BASE[2, 3] = 0.02932887 m`, `damping=0.6`, `armature=0.028`, `frictionloss=0.052`, `motor_torque_limit_nm=2.94`, and `632.006 g` consolidated CAD link inertials with zero synthetic hub masses).
  - Replaced `mujoco.studio` (which lacks a software OpenGL 4.1 driver on Windows and drops `gfx=""` targets in Filament) and browser dashboards with MuJoCo 4.0.1's native C++ desktop `Simulate` application (`mujoco.viewer.launch_passive(..., show_left_ui=True, show_right_ui=True)`), verified visible via Win32 `IsWindowVisible=True`.
  - Streamlined default CogMan & BC skill pipeline (`plate_spin_sort`): pure-contact top-down `PickBlock` -> `PresentAndSpinAtPlateCamera` (`+338 deg` counter-clockwise sweep of Joint 5 `Revolute_4` in front of the low `"Plate"` `side` camera, capturing `17` valid views while polling anomaly messages/raw distances) -> `SortBlockToPile` (`good` green KALAS plate at `[0.10, -0.22]` when clean; `defective` pink KALAS plate at `[0.10, +0.22]` when an anomaly message arrives; refuses to sort on missing thresholds, detector exceptions, or invalid views). Strict `_forbid_teleport` anti-cheat assertions pass across both branches in `tests/test_block_inspector.py`.
  - Live native MuJoCo 4.0.1 desktop demonstration collection, BC training, and closed-loop rollout (`collect-and-train --part duplo_2x2_3437 --episodes 4 --hold-viewer-seconds 300`): recorded `4` episodes (`1,226` frames + `overhead`/`side`/`wrist` RGB-D keyframes across clean and anomaly-flagged trajectories), trained Fourier-phase `ChunkedMLPPolicy` (`chunk_horizon=8`, `train_mse=0.000231`), and passed live closed-loop MuJoCo evaluation (`closed_loop_eval_sorted_on_plate=true`).

### Limitations and unexecuted hardware/cloud work

- All localization, pure-contact grasping, Joint-5 Plate-camera spin, and KALAS plate sorting metrics above are **synthetic MuJoCo 4.0.1 simulation results**, not physical robot or real defect-detection claims. Any anomaly score hooks use raw distances, never probabilities.
- Live bench execution on the physical Intel cell (`armd` follower, RealSense D415s, Sonix gripper camera), VLM target selection / Qualia cloud checks, WendyOS container builds, Intel GPU/NPU/XPU inference benchmarks, and Modal A100/H100 training jobs were not executed in this local task.

## Real-time lightweight Real2Sim point-cloud pipeline & closed-loop execution (Windows local) — 14 November 2026

Extracted teammates' 15 depth-perception and multi-camera fusion modules from `origin/line-mark-test` (`fd8de45`) and `origin/main` (`554020e`) (`scripts/live_depth.py`, `scripts/realsense_depth_capture.py`, `scripts/merge_brick_preview.py`, `scripts/board_alignment_pilot.py`, `scripts/align_pose_preview.py`, `scripts/compare_depth_scans.py`, `scripts/patchcore_depth_viewer.py`, `scripts/detect_brick_line.py`, `scripts/orange_corners.py`, `scripts/reconstruction_audit.py`, `scripts/reconstruction_review.html`, `configs/rgbd-viewer.json`, `src/brickcell/reconstruction.py`, `src/brickcell/inspection/rgbd.py`, `src/brickcell/inspection/line_reference.py`) together with `assets/duplo/stable_poses.json` from `origin/sim/station-twin`.

Implemented `src/brickcell/block_inspector/real2sim.py` (`Real2SimPipeline`, `Real2SimSceneTracker`, `perceive_bricks_and_reject_clutter`, `localize_and_subtract_support_surfaces`, `filter_robot_arm_points`, `ingest_teammate_depth_packet`, `fuse_teammate_top_and_plate_packets`, `decode_live_depth_snapshot`, `load_point_cloud_file`, `save_point_cloud_ply`, `synthesize_cluttered_scene_point_cloud`) and wired it into `SO101InspectionTwin` (`sync_real2sim_scene`, `initial_T_world_block`, `plate_positions`, and `viewer.user_scn` distractor overlay) and the `python -m brickcell.block_inspector real2sim` CLI subcommand.

### Executed checks and simulation evidence

- **Multi-source ingestion**: Tested ingestion of Dual-D415 RGB-D frames, teammate `live_depth.py` / `realsense_depth_capture.py` JSON packets (`top-points.json`, `plate-points.json`, `provisional-transform.json`), `/snapshot/{top,plate}` base64 JPEG + 16-bit PNG packets, ASCII `.ply`, and RealSense SDK `binary_little_endian` `.ply` files in viewer coordinates `(x, -y, -z)`.
- **5-Gate distractor rejection & 6D asset localization** (`python -m brickcell.block_inspector real2sim --perceive-only --headless --fast --num-blocks 4`):
  - Localized all 4 official LEGO DUPLO brick assets (`duplo_2x2_3437`, `duplo_2x4_3011`, `duplo_1x2x2_76371`, `duplo_2x3_curved_top_2302`) across their `stable_poses.json` resting poses with `1.15–2.64 mm` surface ICP RMSE (`0.986–1.000` fitness, `< 0.20 mm` mean centroid replication error) and localized the 3 IKEA KALAS plates (`source`, `good`, `defective`) in `~129–180 ms` per 94,002-point fused cloud on CPU.
  - Filtered out and ignored 100% of injected non-target distractors: `human_arm` (463-point reaching forearm/hand cluster), `wire` (2 tabletop USB/power cable clusters, 123 & 145 points), `trash` (3 clusters: crumpled white/grey cup, brown cardboard scrap, and orange calibration board fragment), and `robot_arm` (`3,162` points masked via calibrated SO-101 forward kinematics and camera housing OBBs).
- **Closed-loop MuJoCo 4.0.1 Desktop replication & policy/planner execution**:
  - `CogManInspectionAgent` closed-loop run (`real2sim --fast --num-blocks 2 --controller cogman_planner`): replicated the perceived bricks and plates in MuJoCo 4.0.1 Desktop (`IsWindowVisible=True`, `mean_replication_pos_err_mm=0.17`), picked the perceived brick via pure contact physics, spun Joint 5 counter-clockwise in front of the low Plate camera, detected the surface defect blemish (`surface_blemish_detected_in_plate_cam`), and sorted it into the pink `defective` KALAS plate (`plate_xy_err_mm=8.06`).
  - `ChunkedMLPPolicy` closed-loop run (`real2sim --headless --fast --num-blocks 1 --controller chunked_mlp_policy`): replicated the perceived scene (`mean_replication_pos_err_mm=0.13`), trained an action-chunked behavioral cloning policy (`policy_train_mse=0.0011254`), reset to the perceived initial state, and rolled out the learned policy closed-loop into the target KALAS plate (`execution_succeeded=true`).
- **Automated test suite & linting**:
  - `tests/test_block_inspector.py -k "real2sim"`: **3 passed** in 20.57s (`test_real2sim_distractor_rejection_and_6d_brick_localization`, `test_real2sim_teammate_depth_packets_and_ply_ingestion`, `test_real2sim_closed_loop_replication_and_policy_rollout`).
  - `ruff check src tests scripts` and `ruff format --check src tests scripts` passed across all 79 Python files.

### Limitations

- All point-cloud synthesis, distractor injection, MuJoCo 4.0.1 Desktop scene replication, and closed-loop policy/planner executions above were run locally in simulation on Windows (`AMD64` CPU); no physical robot commands, SSH connections, or live RealSense hardware captures were performed during this task.

## Three-plate 8-block cluttered simulation & complete pre-hardware architecture (Windows local) — 14 November 2026

Extended the simulated cell to three IKEA KALAS plates (`source` aquamarine blue `[0.28, 0.82, 0.80]` at `[0.23, 0.00, 0.0]`, `good` pastel green `[0.62, 0.86, 0.76]` at `[0.10, -0.22, 0.0]`, and `defective` pastel pink `[0.90, 0.72, 0.78]` at `[0.10, +0.22, 0.0]`) populated with 8 randomly generated mixed DUPLO blocks (`duplo_2x2_3437`, `duplo_2x4_3011`, `duplo_1x2x2_76371`, `duplo_2x3_curved_top_2302`) settled under MuJoCo 4.0.1 contact physics inside the aquamarine `source` plate, and completed all remaining simulation-capable architecture layers (`openvino_anomaly.py`, `cell_adapter.py`, `runtime_nodes.py`, `services/*`, `policy_skills.py`, `cloud/modal_app.py`).

### Executed checks and simulation evidence

- **Unified repository pytest suite** (`artifacts\arch-venv\Scripts\pytest.exe -q`): **83 passed, 5 skipped, 0 failed** in 306.41s (`69` core Brickcell tests + `14` end-to-end tests in `tests/test_block_inspector.py`; the `5` skips remain the opt-in `BRICKCELL_MODEL_CONFIG` and `BRICKCELL_SIMULATION_URL` markers).
- **Lint, formatting & manifests**:
  - `artifacts\arch-venv\Scripts\ruff.exe check src tests scripts services cloud` -> `All checks passed!`.
  - `artifacts\arch-venv\Scripts\ruff.exe format --check src tests scripts` -> `79 files already formatted`.
  - `artifacts\arch-venv\Scripts\python.exe -m brickcell.block_inspector validate` -> exit code 0 (`wendy_manifest_valid=true`, `qualia_manifest_valid=true`, `duplo_assets_verified=true`).
- **OpenVINO 2026.4.1 CPU Plate-camera anomaly detector (`openvino_anomaly.py` + `OpenVINODetector`)**:
  - Explicit leakage-checked partitions rendered from MuJoCo 3D Plate-camera (`realsense_2`) depth-isolated crops with disjoint `object_id` and `session_id` groups: `train/good: 32` (trained strictly on `good` views), `val/good: 8`, `val/defective: 8`, `test/good: 8`, `test/defective: 8`.
  - Exported OpenVINO 2026.4.1 IR (`plate_anomaly_detector.xml`/`.bin`/`.json`, `model_version = plate-patch-ov-8e1a4280c747`), calibrated raw patch-distance threshold on `val` (`threshold_value = 0.520599`, `method = validation_max_f1_ties_favor_recall`), and evaluated once on held-out `test`: **`TP=8, TN=8, FP=0, FN=0`** (`test_min_raw_distance = 0.3055`, `test_max_raw_distance = 0.7417`). All scores are raw patch distances, never probabilities.
- **3-plate 8-block cluttered singulation & OpenVINO sorting (`clear_source_plate`)**:
  - Verified in `test_three_plates_eight_clustered_mixed_duplo_blocks_openvino_sort` with `_forbid_teleport` anti-cheat guard active: segmented all 8 clustered mixed DUPLO bricks in `plate_source` (with conical KALAS rim masking and `stride=1` empty-plate verification), peeled the cluster outside-in via `select_clutter_safe_topdown_grasp`, picked **8/8** bricks purely by jaw contact friction, spun Joint 5 (`Revolute_4`) counter-clockwise in front of the low Plate camera, scored depth-isolated foreground crops via `OpenVINODetector`, and sorted **8/8** bricks (`4` clean -> `plate_good`, `4` synthetic-blemish -> `plate_defective`) across non-overlapping `PLATE_DROP_SLOT_OFFSETS_XY` ring slots via a `z = 16.5 cm` center-plate clearance waypoint.
- **Core `Supervisor` boundaries & multi-node BabyROS / XPolicyLab integration**:
  - `test_core_supervisor_boundaries_and_openvino_manifest_partitions`: ran `brickcell.controller.supervisor.Supervisor` end-to-end over `TwinRobotAdapter(RobotInterface)` (`mode = "mujoco_simulation_not_physical"`, non-blocking `ActionStatus.PENDING -> SUCCEEDED` slices), `TwinPlateFrameSource(FrameSource)`, and `InspectionService(OpenVINODetector)` (`completed_cycles = 1`, `errors = 0`, `physical_success_verified = False`).
  - `test_multinode_babyros_runtime_graph_and_xpolicylab_checkpoint_roundtrip`: verified closed-loop multi-node communication across `CameraPubNode -> PerceptionAndAnomalyNode -> FraxMujocoTwinNode <-> CogManInspectorNode` over `InProcessRuntimeBus` (`so101/inspection/anomaly` raw-distance alerts; detector never commands robot), `ChunkedMLPPolicy` `.npz` checkpoint round-trip, `lerobot_v3` `meta/info.json` + SHA-256 `kept_episodes_manifest.json` export, and `RemoteXPolicyClient` <-> `LocalXPolicyWebSocketServer` msgpack-numpy action-chunk round-trip plus stationary-hold timeout/unreachable safety fallback.

### Limitations and unexecuted hardware/cloud work

- All 8-block clutter perception, contact-friction grasps, OpenVINO raw-distance evaluations, `Supervisor` cycles, and multi-node BabyROS runs above use **synthetic MuJoCo 4.0.1 simulation data**, not physical defect-detection accuracy, physical grasp reliability, or Intel GPU/NPU/XPU hardware benchmarks.
- Physical bench connection (`ssh.md` / `hardware.md` / `arm-api.md`), real STS3215 servo motion, physical RealSense D415 / Sonix camera capture, and remote cloud training/inference were intentionally not executed prior to starting the hardware phase.

## Live station twin dashboard — 14 November 2026

`dashboard/` adds a read-only page (user service `twin-dashboard`, Intel `127.0.0.1:8096`):
the Top camera as MuJoCo sim render or real D415 frames, the fused Top + Plate D415 point
cloud in the robot base frame, and focus-block / anomaly overlays. See `dashboard/README.md`.

Executed:

- `pytest dashboard/tests` (uv, Python 3.12, numpy, OpenCV headless): 13 passed. Fake
  upstream services; covers camera-to-world maths, packing, crop, table levelling, overlay
  validation/expiry, stale depth, a dead source, frame long-poll, gzip cloud, MJPEG.
- `ruff check dashboard` and `ruff format --check dashboard`: passed.
- Deployed to Intel with `dashboard/deploy.sh`. Only the new service was added; the live depth
  service (8094), the sim server (18081/19091/19092), armd and camui were not changed.
- Live end-to-end in a browser through an SSH tunnel: real Top frames, sim `overhead` renders
  and both D415 clouds (about 14k points each) updated. Measured over Wi-Fi: sim 3.9 fps
  (source 4), real 5.5 fps (source 12, tunnel-limited), cloud 2.1 Hz, first frame 84 ms after
  `T`. Checked sim/real toggle, 1×/2× zoom, full screen and `Esc`, depth presets
  (`0 7 1 3`, `Ctrl`), orthographic `5`, drag orbit/pan, wheel zoom, gizmo click and the
  bottom-left tool buttons.
- Overlay stubs: posted labelled TEST boxes to `/api/anomalies` and `/api/focus`; the page drew
  them, *Focus only* filtered to one, and they expired 5 s after the posts stopped.
- Failure paths with a second local instance: a dead real source showed *No signal* with the
  error and *Degraded*; a stopped server showed *Server unreachable* and removed all overlays;
  the page recovered without reload.

Found and limits:

- The sim scene camera poses no longer match the table: the live table fit gives about 4° tilt
  for Top and about 7° and 5 cm for Plate. The dashboard levels each cloud on the table, but the
  Top and Plate clouds still disagree horizontally (duplicate plate rims), and the sim render
  uses the same stale poses. A new camera calibration is needed.
- The sim is not twin-synced; its bricks and arm do not follow the real station.
- Focus block and anomaly detection are not integrated; the page shows *Not connected*.
- Not measured: frame rate on the Intel display itself (no tunnel), or a long-run soak test.

## Plate + wrist RGB mark training on main — 15 November 2026

Switched to main, preserved its pre-update history as
`backup/main-before-plate-wrist-20261115` and merged current origin/main without
resetting either history. Resolved viewer conflicts in favor of the newer upstream
implementation and retained both validation histories. No changes were pushed.

Selected plate and wrist RGB only for the default mark viewer and real-camera
anomaly inspector. Top remains available for scene localization, but is excluded
from mark scoring; depth models are excluded before compilation and depth is not
passed into the RGB scorer. Uncalibrated Lab heuristics cannot become trusted
GOOD/DEFECTIVE results. No robot commands, camera access or hardware deployment
were executed. The old explicit RGB/depth review configuration remains available.

Added joint preparation for plate/wrist datasets. It rejects duplicate source/
crop hashes, object/session partition overlap across cameras, defective training
images and missing camera/type train/val/test classes. Both cameras use the same
256x256 ResNet-18/layer2+layer3 PatchCore architecture, with separate memory banks,
camera/crop contracts and validation thresholds. Training and checkpoint reload
verify those contracts; existing no-contract models remain backward-compatible
when their dataset/config also has no contract.

Executed actual Anomalib 2.7.0 CPU reference training for plate camera on six
accepted clean brick-002 crops from the same source object/session, then reloaded
the checkpoint and scored six later clean frames. Saved source/record/crop/model
hashes, exact versions, raw-distance maps and comparison JSON under ignored
`artifacts/pilots/plate-marks-clean-v1/`. Input crop `[332,30,414,104]`, model input
256x256, 5% coreset (306 features). Clean-repeat distances: 1.0930704–1.2688699.
No threshold or independent test accuracy was estimated. Resizing this small
source crop does not create additional detail. The original capture's formal
training eligibility remains unchanged. The pilot is not installed into any
live model slot and cannot authorize sorting.

The saved defect was confirmed facing the top camera, with changed plate framing;
it was not relabeled as verified plate defect evidence. No labeled wrist dataset
was found locally (only a saved wrist preview). Production wrist fitting and
independent plate/wrist validation remain pending new labeled captures. New
collection must preserve physical brick/session groups and confirm each mark is
visible in its labeled camera image. No strong mark-recall claim is made.

Executed `uv sync --locked`; `uv run pytest`: **184 passed, 16 skipped**. Optional
skips require FastAPI/Frax/JAX/MuJoCo, compatible trained/exported core assets or
the live simulator. Nine new tests cover joint leakage, duplicates, defective
train rows, missing cameras, excluded top views, camera-contract swaps and actual
RGB-only inspector/transport behavior. The inspector boundary test executes its
actual classes without importing optional robot SDKs. A prior MuJoCo eager-import
collection failure was resolved by deferring simulation imports to operations
that use them; the upstream mixed architecture test now declares missing JAX/Frax
as optional, consistent with adjacent simulation tests. No optional simulator or
accelerator dependencies were installed in core. Ruff lint and formatting passed
for all 114 files in src/tests/scripts; whitespace checks passed. Formal model
export/parity and independent test scoring were not performed for this pilot.
See `docs/PLATE_WRIST_MARKS.md` for actual commands and the remaining data gap.

### Recall-constrained plate/wrist calibration (2026-11-15)

New mark preparation configs require validation defect detection recall 1.0 and
at most 0.1 clean nonaccept rate (DEFECTIVE plus REINSPECT). The policy evaluates
actual threshold decisions including the uncertainty margin; uncertain defects
do not count as detections. Infeasible constraints raise an error without saving
approved calibration metadata. Existing configs retain F1 calibration by default.
Validation metadata and held-out reports now count actual decisions separately
from binary metrics. Test reports include brick type, face, object and session
breakdowns, preserving that repeated images are not independent trials.

Executed `uv sync --locked`, 27 focused checks, then `uv run pytest`: **193 passed,
16 skipped**. The nine new calibration/report tests use synthetic score arrays;
one demonstrates that constrained calibration catches a low-score defect that
max-F1 would accept as good, and others reject overlapping distributions or
excessive clean reinspection. They are software-policy checks, not measured
camera performance. Optional FastAPI/Frax/MuJoCo/JAX checks, four model tests
requiring compatible trained/exported assets, and live simulator checks remain
skipped. Ruff lint/format passed for 115 files; `git diff --check` passed.

No new real model was fitted, exported or deployed in this change. Local data
still lacks independent plate mark evaluation and labeled wrist training data.
The earlier clean-only reference cannot establish defect recall or calibrate the
new policy. A 100% validation target is not a claim of 100% future detection;
real camera-specific calibration and untouched test evaluation remain pending.

### Required plate/wrist scan evidence (2026-11-15)

Runtime review found that the presentation scan could complete from three valid
plate frames without usable wrist results. The scan now requires both configured
models before motion, resets prior completion, and records the seven scheduled
plate and three scheduled wrist observations. Clean completion requires every
scheduled view to be valid; missing or uncertain wrist evidence prevents it.
Repeated camera/frame identities cannot stand in for new observations. Trusted
defect flags persist across views. Existing capture/inspector failures still
abort. Top remains used for localization/display but is no longer submitted for
anomaly scoring during the scan.

Executed seven new software tests, including execution of the actual scan method's
missing-model preflight without importing robot SDKs. Full `uv run pytest`:
**200 passed, 16 skipped**; Ruff lint and formatting passed (117 files), and
whitespace checks passed. The optional physical-cell mock test was updated to
expect refusal to sort from the uncalibrated fallback; it remains skipped because
MuJoCo is absent. No robot/camera was accessed, and no source changes were deployed.
Scheduled-view completeness is not proof of physical face coverage, visibility
of tiny marks or real-world defect accuracy. The same labeled-data gap remains.

### Block-surface PatchCore input and reference experiments (2026-11-15)

Implemented optional versioned colored-block surface preprocessing, shared by
the explicit training dataset, PyTorch scoring, reference pilot and OpenVINO
runtime. The silhouette retains enclosed dark/gray defects, replaces outside
pixels with fixed gray, and applies bounded global exposure normalization. Raw
PatchCore map distances are scored only inside that mask. Camera/manifest/model
profile checks prevent accidental preprocessing swaps. Existing models without
a profile retain their original path. This feature requires retraining and new
validation calibration; no old threshold was reused with the changed statistic.
Surface-specific presence, ambiguity, clipping, native size, exposure, glare and
sharpness checks yield an untrusted view/REINSPECT instead of a defect decision.
The inspector still cannot command hardware.

Executed `uv sync --locked`. New tests cover changed exterior pixels, retained
black/gray marks, bounded exposure changes, ambiguous/clipped/missing objects,
outside-map hotspots, actual dataset/PyTorch/OpenVINO adapter agreement using
injected model outputs, and untrusted-view behavior. Final `uv run pytest`:
**216 passed, 16 skipped**. Skips remain optional FastAPI/Frax/MuJoCo/JAX, four
formal model tests needing calibrated/exported assets, and the live simulator.
Ruff lint/format passed (120 files); whitespace checks passed. No dependencies
were added or downgraded.

Actual CPU Anomalib 2.7.0 reference training:

- `artifacts/pilots/plate-marks-surface-v1/`: six accepted clean blue plate crops
  fit at 256x256, ResNet-18 layer2/layer3, 306-entry coreset; six later clean frames
  scored after checkpoint reload. Source crop remains only 82x74 pixels.
- `artifacts/pilots/red-marks-surface-v1/`: one existing accepted red reference,
  same architecture, 50-entry coreset. Clean repeat distance 2.397419; existing
  gray-patch image 3.110886. The marked image's peak is at source (33.1,7.1), near
  the studs rather than the gray patch. Higher image score does not establish
  correct mark localization.
- `artifacts/pilots/red-marks-surface-fine-v1/`: separate layer1/layer2 experiment,
  203-entry coreset; clean repeat 2.146855 and marked image 3.285108. Its peak also
  remained near the studs (36.6,10.0); not adopted as an established improvement.

Source image/checkpoint hashes, object/session IDs and exact package versions
are preserved. All pilots remain uncalibrated, reference-only and prohibited
from automatic sorting. Original capture eligibility was not changed. The red
development images must not later be counted as independent held-out tests.

Executed `scripts/audit_surface_pilot.py` against one blue clean-repeat image.
Three **synthetic** background substitutions (black, white, warm brown) produced
identical model input tensors and score 1.370452. Synthetic global exposure and
smooth-gradient shadow probes changed score by at most 0.010163. Synthetic black
line probes, width 1/2/4 source pixels, scored 3.048703/3.384203/3.366593. No
threshold was selected from those probes; they do not measure real defect recall
or arbitrary environmental invariance. Machine report:
`artifacts/pilots/plate-marks-surface-v1/robustness/report.json`.

Exported the blue reference checkpoint to uncompressed OpenVINO IR for a native
CPU parity diagnostic on six clean repeats. Maximum absolute surface-score
difference: **0.0000025034**, below 0.001. Report and model hashes:
`artifacts/pilots/plate-marks-surface-v1/openvino-reference-check/parity.json`.
The export invocation from stdin emitted child-process `<stdin>` import warnings
but completed successfully, wrote IR, ran CPU inference and passed numerical
parity. This is not threshold/decision parity or a calibrated deployment artifact.

Reviewed mask previews and verified all 15 selected existing blue/red reference
crops pass the final surface quality checks. These profiles cover the inspected
colored crops only. White/black blocks, edge marks, occlusion, similar-colored
backgrounds, strong local shadows/glare and unseen poses are not validated. No
labeled wrist training set or independent real marked plate test set was found.
No camera, robot, SSH service or cloud account was accessed, no trained artifact
was committed, and no live viewer/model slot was changed. Further real data and
validation are required before claiming reliable mark-only acceptance/rejection.

### Foreground memory and corrected dark-background mask (2026-11-15)

Added opt-in `foreground_memory` training. `SurfacePatchcore` reuses Anomalib
2.7.0's feature extraction, pooling, embedding construction and coreset selection,
but only stores feature cells with >=50% foreground-mask occupancy. Background
replacement pixels therefore cannot directly contribute their own patch cells to
the normal bank. This does not eliminate background from CNN receptive fields.
The exact training mask is reconstructed from the source image using the same
fixed preprocessing; anomaly ground-truth masks are not repurposed. Only explicit
good labels are accepted. Config/checkpoint metadata preserve the selection policy
and kept/candidate counts; ordinary Patchcore reloads the weights for inference.

Executed CPU fits and checkpoint reloads for multiple development candidates:
foreground-only memory with the previous red mask, two legacy clean views,
layer1-only features, and the final corrected-mask candidate. An unblurred-map
diagnostic was also run without modifying saved checkpoints. Neither foreground
selection alone, extra clean view, layer1 alone nor removing blur resolved the
false peak. The second legacy clean reference lacks recorded physical-session
identity; its multiview artifact explicitly records incomplete source metadata
and is ineligible for calibration or independent accuracy claims.

Pixel review localized the false peak to nearly black background above the studs
with a red color cast. Updating the red example surface profile's minimum seed
value from 25 to 40 excludes those exterior pixels while retaining enclosed dark
marks. The new `artifacts/pilots/red-marks-foreground-mask-v2/` fit used the same
single accepted clean reference, 256x256 ResNet-18 layer2/layer3, 571/1024 eligible
foreground feature cells and a 27-entry coreset. After reload, the marked query's
peak moved from `(33.1,7.1)` to **`(48.4,64.6)` on the gray patch**. New raw
distances: clean repeat **2.498976**, marked **3.018789**. The clean maximum still
occurs near a boundary. This demonstrates better localization on one development
example, not improved independent recall or a calibrated clean/marked classifier.
All compared examples are development data and cannot later serve as untouched
test images. No threshold was fitted or model deployed.

`scripts/check_reference_export.py` exported the corrected candidate and ran
native OpenVINO CPU parity on both development queries: **passed**, maximum
absolute score difference **0.0000014305**. The guarded file-based export avoided
the earlier stdin child-import issue. IR/checkpoint/source hashes and exact
versions are recorded under the candidate's `export-check/parity.json`.

`scripts/audit_surface_pilot.py` on its clean repeat: three synthetic background
replacements gave identical input tensors and scores (2.498976); exposure probes
changed distance by at most 0.034254; a synthetic smooth shadow changed it by
0.115145. Synthetic 1/2/4-pixel line probes scored 2.886249/3.303150/3.450163.
These are controlled diagnostics, not real-world environmental invariance or
real marker-line detection rates. `localization-review.png` was rendered and
visually checked, with both clean and marked examples and equal heatmap scales.

Full `uv run pytest`: **223 passed, 16 skipped**, two upstream deprecation warnings.
Seven added tests cover foreground embedding selection/order/occupancy, per-image
nonempty masks, required surface contracts, prepared-config policy propagation,
and exclusion of a dark red-cast background while preserving an enclosed black
mark. Ruff lint/format passed (123 files after adding the executed export checker),
and `git diff --check` passed. The same optional hardware/simulator/model skips
remain. No robot, SSH server, camera or cloud account was accessed. Independent
plate validation/test images and labeled wrist captures remain unavailable.

### Camera connection restored (2026-11-15)

At the user's explicit request, followed `ssh.md` and `hardware.md` to restore
camera access. The local `intel` SSH alias was missing. Restored it using the
existing local Intel key and the documented pinned host keys with strict checking;
verified remote identity `intel` / `intel-XPS-16-DA16260`. No private key material
was printed or added to the repository. The remote camera services were already
running; no remote service, driver or robot-control changes were needed.

Restored loopback-only forwards from local 18094 to Intel 8094 and local 18091 to
Intel 8091. Verified two distinct plate frame IDs with real serial `243622060450`,
640x480 decoded RGB, `synthetic=false`, and reported ages 0.030/0.057 seconds.
The wrist endpoint returned a valid 1920x1080 JPEG. It still lacks acquisition
time, frame-ID and camera-serial headers, so wrist freshness is not established
and that feed remains preview-only for trusted inspection. Local viewers on
18094 and 18097 returned HTTP 200. No calibrated anomaly model was installed,
no classification accuracy was measured, and no robot motion was issued.

### Live user brick 1 capture (2026-11-15 local time)

At the user's request, captured six fresh plate-camera RGB images after they
placed brick 1 in view. First/last frames show a white rectangular block on the
blue plate with a clear black X on its camera-facing wall. This is a visually
marked example, not a clean reference. Stored under ignored
`artifacts/live-marks-20261114T235451Z/brick-001/plate-marked/` (UTC folder time),
with capture timestamps, distinct frame IDs, SDK serial `243622060450` and image
hashes. The record distinguishes assistant visual annotation from the user's
placement statement, leaves `human_label` unset, and marks clean-training
eligibility false. Physical identity across earlier sessions is unverified.

The existing red/blue reference models are not calibrated for this white block;
no model pass/fail result was produced and no defective image was fitted as good.
A matching unmarked reference is needed next. The gripper changed position in the
background between the first/last images while the marked block stayed visible;
no motion command was issued by this assistant. No software logic or remote
service was modified.

### Live clean white reference pilot (2026-11-15)

The user confirmed an unmarked matching white brick. Captured twelve plate RGB
frames, with frame IDs, acquisition metadata and hashes under ignored
`artifacts/live-marks-20261114T235451Z/white-clean-reference/plate-clean-235841/`.
Reviewed crop `[212,220,344,320]`; all twelve passed geometric surface checks.
Added an opt-in experimental pale-block profile for this saturated blue backdrop.
It is not a general white-block segmentation contract or a deployed model.

Fit CPU PatchCore on six clean frames only, using foreground memory. The six
remaining same-object, same-session clean frames scored raw distances
1.870613–2.068170. This is a development reference repeat, not an independent
validation/test partition or an accuracy measurement. Model, source hashes and
package versions are recorded in `artifacts/pilots/live-white-clean-20261115-v1/`.
No marked images were fitted; threshold remains unset and sorting disabled.

Visual review of the separate marked comparison's `surface-preview.png` found
that the mask excludes part of the X where it meets the lower block silhouette.
The geometric checks did not catch this limitation. Recorded REINSPECT in the
local comparison review and withheld a trusted mark decision. Surface retention
must be repaired and validated before this profile can support mark-detection
claims. Neither clean-repeat scores nor passing software tests establish that
the white pilot detects defects. No robot command or remote deployment occurred.

Verification: `uv run pytest` **225 passed, 16 skipped**, two upstream deprecation
warnings. Skips cover unavailable optional hardware/simulation dependencies,
four configured-model tests, and the live simulator integration. Ruff check and
format check passed (123 files). The two added pale-profile tests verify enclosed
mark retention on synthetic imagery and reject ambiguous profile settings; they
do not cover the observed real silhouette-touching X failure.

### Additional clean white face capture (2026-11-15)

After the user rotated the clean white brick and said ready, captured six distinct
fresh RGB frames from the existing plate snapshot service (SDK serial
`243622060450`, synthetic false). Saved original images, hashes and acquisition
metadata under ignored `artifacts/live-marks-20261114T235451Z/white-clean-reference/plate-clean-view-02-001948/`.
First/last visual review shows the eight-stud face toward the camera, a smooth
wall above, full block visibility and no visible marker mark or hand occlusion.
Recorded user-supplied good label, same object/session group and suggested crop
`[211,233,339,319]`. Accepted as clean reference capture only; not yet fitted or
assigned to a formal dataset partition. No inference decision, robot command,
deployment or software change was made. The mask limitation recorded above
remains unresolved.

### Clean white short-end capture (2026-11-15)

Captured six distinct fresh real plate RGB frames after the user's ready signal,
stored with hashes and acquisition metadata under ignored
`artifacts/live-marks-20261114T235451Z/white-clean-reference/plate-clean-view-04-002313/`.
First/last visual review confirms a short end facing the camera with the studded
upper surface visible, full block visibility and no visible marker mark or hand
occlusion. Recorded the same user-labeled clean object/session, suggested crop
`[232,226,312,337]` and clean-reference acceptance. These frames are collected,
not fitted or formally partitioned. No inference decision, robot command,
deployment or software change. The recorded surface-mask limitation remains.

### Additional angled clean white capture (2026-11-15)

Saved six distinct fresh real plate RGB frames with hashes and acquisition
metadata under ignored
`artifacts/live-marks-20261114T235451Z/white-clean-reference/plate-clean-view-05-002614/`.
First/last review shows a fully visible clean white brick at an angle, exposing
a short end, long side and studded upper surface. A person's arm/hand is outside
the suggested block crop `[158,220,280,333]`, without block occlusion. Symmetry
prevents verifying that the opposite physical end was exposed. Recorded the
same user-labeled clean object/session; accepted as reference capture only,
not fitted or formally partitioned. No model decision, robot command, deployment
or software change. The surface-mask limitation remains unresolved.

### Combined live white clean reference fit (2026-11-15)

Captured six distinct fresh real plate frames after the next ready signal at
`artifacts/live-marks-20261114T235451Z/white-clean-reference/plate-clean-view-06-002657/`.
First/last review shows the hollow underside facing the camera with a smooth
wall above, rather than verified opposite-long-side coverage. Full brick visible,
no hand occlusion or visible marker. Recorded crop `[176,214,305,313]`.

Prepared all six accepted clean capture groups with reviewed per-group crops,
checking original hashes and rejecting duplicate source bytes. Six images from
one group failed the existing clipped-glare limit and were excluded without
relaxing that limit. CPU PatchCore fitting completed on **36 clean images from
five capture groups**, using the experimental pale-block foreground profile.
Stored per-image provenance, rejection reasons, checkpoint hash and package
versions in ignored `artifacts/pilots/live-white-clean-multiview-20261115-v2/`.
The earlier v1 preparation stopped on the glare rejection and fitted no model.

All inputs belong to the same clean physical object/session. No validation/test
partition or accuracy evaluation was performed; the combined model has no
threshold and cannot authorize sorting. Marked images were never fitted.
The known mask failure on a silhouette-touching X remains unresolved, and
additional clean views do not fix or validate it. No robot command, model
deployment or tracked software change occurred. This entry records executed
CPU reference fitting, not production readiness or complete physical coverage.

### Next-sized clean block capture (2026-11-15)

Captured six distinct fresh real plate RGB frames after the user's next-block
ready signal. Stored hashes and acquisition metadata under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-002/plate-view-01-002953/`.
First/last review shows a blue 2x2 four-stud block, front wall and upper surface
visible, without hand occlusion or visible marker. Recorded a new object ID,
same capture-session leakage group, user-supplied clean label and suggested crop
`[212,239,291,323]`. Accepted as raw clean reference only, not fitted or formally
partitioned. Blue block against blue plate needs foreground validation; the white
profile is not applicable. No inference decision, robot command, deployment or
software change occurred.

### Blue 2x2 studded-face capture (2026-11-15)

Saved six distinct fresh real plate RGB frames with hashes and acquisition
metadata under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-002/plate-view-02-003203/`.
First/last review shows four protruding studs facing the camera, with a smooth
wall above; this does not provide the requested hollow-underside view. Full
block visible, no visible marker or hand occlusion. Recorded same clean object
and session, crop `[232,234,313,323]`, and accepted clean-reference-only status.
Not fitted or formally partitioned. No inference decision, robot command,
deployment or software change.

### Blue 2x2 hollow-bottom capture (2026-11-15)

Captured six distinct fresh real plate RGB frames after the user's ready signal,
stored with hashes and acquisition metadata under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-002/plate-view-03-003244/`.
First/last review confirms the hollow underside and central tube facing the
camera, full block visibility and no hand occlusion or visible marker. The
interior is dark; small-mark visibility there is unverified. Recorded same
user-labeled clean object/session and crop `[232,233,313,320]`. Accepted clean
reference capture only; not fitted, formally partitioned or evaluated. No robot
command, inference decision, deployment or software change occurred.

### Blue 2x2 smooth-side capture (2026-11-15)

Saved six distinct fresh real plate RGB frames with hashes and acquisition
metadata under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-002/plate-view-04-003326/`.
First/last review confirms a smooth side facing the camera, with hollow underside
upward rather than studs. Full block visible; no visible marker or hand occlusion.
Symmetry prevents identifying which physical side is exposed. Recorded same
user-labeled clean object/session, crop `[257,233,335,316]`, and clean-reference
acceptance only. These frames remain collected, not fitted or evaluated.
No robot command, deployment, inference decision or software change occurred.

### Third clean block initial capture (2026-11-15)

At the user's next-block ready signal, saved six distinct fresh real plate RGB
frames with acquisition metadata and hashes under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-003/plate-view-01-003410/`.
First/last visual review shows a yellow tall rectangular block with two upper
studs visible and a broad smooth wall toward the camera, without hand occlusion
or visible marker. Recorded a separate object ID with the same session leakage
group, clean label from the ongoing user clean-capture workflow and suggested
crop `[219,224,303,319]`. Accepted as clean reference capture only; not fitted,
formally partitioned or evaluated. No robot command, deployment, inference
decision or software change occurred.

### Yellow block hollow-bottom capture (2026-11-15)

Saved six distinct fresh real plate RGB frames with acquisition metadata and
hashes under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-003/plate-view-02-003453/`.
First/last review shows a hollow bottom opening and inner tube facing the camera
at a shallow angle beneath a broad smooth wall. Complete block visible, without
visible marker or hand occlusion; full interior visibility is unverified.
Recorded same user-labeled clean object/session, crop `[237,228,313,303]` and
accepted clean-reference-only status. Collected, not fitted or evaluated.
No robot command, inference decision, deployment or software change occurred.

### Yellow block narrow-side capture (2026-11-15)

Saved six distinct fresh real plate RGB frames with hashes and acquisition
metadata under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-003/plate-view-03-003719/`.
First/last review shows the block lying on its side, smooth narrow wall toward
the camera, broad smooth wall above and studs pointing right. Entire block
visible without hand occlusion or visible marker. Recorded same user-labeled
clean object/session, crop `[211,239,309,322]`, clean-reference acceptance only.
Not fitted, formally partitioned or evaluated. No robot command, inference
decision, deployment or software change occurred.

### Yellow block additional broad-side capture (2026-11-15)

Captured six distinct fresh real plate RGB frames, saved with hashes and
acquisition metadata under ignored
`artifacts/live-marks-20261114T235451Z/clean-block-003/plate-view-04-003756/`.
First/last review shows a broad smooth wall facing the camera, narrow wall above
and studs pointing right. Full block visible without visible marker or hand
occlusion; opposite physical side identity is unverified due symmetry. Recorded
same user-labeled clean object/session, crop `[221,221,318,313]` and reference-only
acceptance. Total yellow captures: 24 images across four poses, collected but
not fitted or formally evaluated. No robot command, deployment, inference
decision or software change occurred.

### Clean white underside capture (2026-11-15)

Captured six additional distinct, fresh, real plate RGB frames after the user's
ready signal. Saved under ignored
`artifacts/live-marks-20261114T235451Z/white-clean-reference/plate-clean-view-03-002049/`,
with hashes, SDK serial and acquisition metadata. First/last visual review shows
the hollow underside obliquely above a smooth long side, with the entire block
visible and no hand occlusion or visible marker mark. This angle does not prove
full visibility inside every cavity. Recorded the same user-labeled clean object
and session, suggested crop `[203,236,331,322]`, and clean-reference acceptance.
Frames are collected, not fitted; no formal partition, accuracy result, robot
command, deployment or software change. The previously recorded mask limitation
remains unresolved.

### Blue/yellow reference fits and first development check (2026-11-15)

At the user's request to start testing, prepared the blue/yellow clean captures
with per-view reviewed crops, source-hash verification and duplicate rejection.
An initial blue profile admitted the plate and failed border checks. The initial
yellow profile cut away the shadowed wall despite passing geometry checks; its
24-image v1 fitted pilot is rejected for live use. The revised blue hue interval
[101,130] and yellow [15,65] retain substantially more visible plastic; reviewed
per-pose surface previews. These are experimental capture-specific profiles,
not validated environmental invariance or full surface retention guarantees.

CPU PatchCore reference fits completed in ignored
`artifacts/pilots/clean-block-002-multiview-20261115-v2/` (12 clean frames,
two poses; 12 excluded for clipped glare) and
`artifacts/pilots/clean-block-003-multiview-20261115-v2/` (24 clean frames,
four poses). Input hashes, exclusions, profiles, checkpoint hashes and package
versions are stored there. No marked images were fitted. The existing white
pilot remains separate; its silhouette-touching mark limitation is unresolved.

Captured a new fresh real plate snapshot of the still-present clean yellow block
and visually reviewed its full visibility. With the same reviewed pose/crop,
its raw distance was 1.589345; a separate explicitly synthetic 3-pixel dark-line
probe scored 3.255861. Results/maps and acquisition metadata are saved under
`artifacts/pilots/live-yellow-development-check-20261115-v1/`. This demonstrates
sensitivity to one artificial line only, not real mark detection, accuracy or a
calibrated pass/fail. The fresh clean query is from the same physical object and
session as fitting and is not an independent validation/test set. No threshold
was selected; sorting remains disabled. Real marked-brick checking is pending.
No robot commands or model deployment occurred. Only ignored experiment scripts
and artifacts plus this documentation were changed; no core software changes.

### Arm connection and inspection preflight (2026-11-15)

At the user's explicit request, read arm-api.md and connected to the existing
armd service through SSH, then added a loopback-only forward on local 18765.
GET health and follower state succeeded: real follower online, six joints online
and healthy with no reported errors. Saved state and preflight evidence under
ignored `artifacts/arm-inspection-preflight-20261115/`. No motion, torque, serial
ownership or remote service change was issued. This establishes connectivity,
not grasping, collision safety or physical task success.

Automatic inspection remains blocked by unset model thresholds, absent trained
wrist model/freshness metadata, and the known silhouette-touching mark masking
limitation. Kept sorting disabled rather than bypassing detector gates. Executed
local scan/controller/inspection/calibration safeguard tests: **46 passed**.
These are software checks, not physical robot or real defect-accuracy tests.

The current plate query visibly contains a yellow block with a black X on a pale
patch. Scored the separate reviewed crop `[233,229,330,321]` using the yellow v2
clean-only model: raw distance **2.756521**, versus earlier clean development
query **1.589345**. Saved query provenance and maps alongside preflight evidence.
This is a same-session development comparison with changed pose/position and a
pale patch as well as a black X; it does not isolate the ink contribution or
establish mark localization, calibrated pass/fail or accuracy. No marked input
was fitted. Arm movement and automatic sorting remain disabled for this test.

### Repeated live marked-yellow diagnostic (2026-11-15)

At the user's request to test immediately, captured three distinct fresh real
plate RGB frames and visually reviewed the first/last: yellow block with a black
X on a pale patch, fully visible. Saved acquisition metadata, source hashes,
reviewed crops `[233,229,330,321]`, maps and results under ignored
`artifacts/live-mark-test-20261115T004253Z/`. The clean-only yellow v2 model gave
raw distances **2.645704, 2.592618, 2.695344**, above the earlier same-session clean
query **1.589345**. Peak positions in crop pixels were (48.88,46.72),
(61.38,42.77), (59.87,42.77). This is repeated-frame development evidence only:
not an independent test, ink-only attribution, accurate mark segmentation or
calibrated pass/fail. No threshold was selected, marked data was not fitted,
and no arm motion or automatic sorting command was issued. The arm inspection
preflight blockers remain. No core software changed.

### Live heatmap review exposes mark masking failure (2026-11-15)

Ran yellow v2 PatchCore on a new fresh plate frame and rendered side-by-side
camera/heatmap visualization under ignored `artifacts/live-algorithm-preview/`.
Raw distance was 2.751703 with crop peak (60,42). Visual review of comparison.png
shows the pale patch and much of the black X are excluded by the foreground
mask, with elevated distances around that missing region and other block edges.
Thus the elevated score does NOT establish that the algorithm detects the ink.
This demonstrates that the revised yellow color contract still fails to retain
this actual mark/patch, despite retaining more clean plastic. Earlier score
separation must not be interpreted as successful mark detection. No calibrated
decision or arm command occurred. Fixing full-surface retention remains required.

### Arm-held screenshot review and connection reacquisition (2026-11-15)

User supplied four depth-viewer screenshots showing an arm-held yellow brick,
including two visible marked faces. These demonstrate presentation views, not
anomaly detector execution or physical actions issued by this assistant. Remote
read-only checks found armd follower released. After checking the serial device
with fuser (no owner reported), reacquired the existing follower bus through
armd. Fresh readings returned six healthy joints, all torque disabled. Elbow
position **3031** was outside configured range **791–2991**; shoulder lift 1196
was near its configured minimum 1195. No motion or torque command was issued,
and configured limits were not modified. Automatic physical execution remains
blocked by this out-of-range starting pose and the previously recorded detector
preflight limitations. Screenshot depth visualization must not be treated as
anomaly recognition or success of an end-to-end sorting cycle.

### Supplied arm-held images: image-only detector test (2026-11-15)

Following the user's image-only instruction, tested all four supplied screenshots
with reviewed plate-region block crops using the existing yellow v2 foreground
contract. Saved source hashes, crops and per-image reports under ignored
`artifacts/screenshot-detection-20261115/`. All four returned REINSPECT with
`Block touches crop border; surface may be clipped` during preprocessing, so no
PatchCore classification was produced. The expanded hue contract also admits
background colors in these arm-held scenes; the border check does not establish
that the actual brick was clipped. Manual review sees an X in screenshots at
02.51.36 and 02.51.49; the X is not visible on the facing surfaces at 02.51.55
and 02.51.59, which does not establish whole-object cleanliness. These results
show that the plate-resting reference pipeline does not transfer directly to
arm-held screenshots. No input was fitted and no arm commands were issued.

### Five supplied white-brick arm-held screenshots (2026-11-15)

Tested the five 02.54 screenshots using the existing white clean-reference pilot
and reviewed plate-region crops. Saved source hashes, crops and results under
ignored `artifacts/white-screenshot-detection-20261115/`. All five returned
REINSPECT from the border-contact foreground check, before PatchCore inference.
The pale-block contract was designed for a saturated blue backdrop and also
admits the pale background in these arm-held views. It cannot establish usable
foreground masks here. User labels are unconfirmed; none were fitted or assigned
formal data partitions. No good/defective result or real defect-accuracy claim
is supported, and no robot commands were issued.

### Anomaly branch snapshot verification (2026-11-15)

Before committing the requested anomaly branch, executed `uv sync --locked`,
`uv run pytest` (225 passed, 16 skipped, two upstream deprecation warnings),
`uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`
(123 files) and staged diff whitespace checks; all completed successfully.
Skips remain optional dependency/hardware/simulator tests and four trained-model
configuration tests. Captures, trained checkpoints, artifacts, credentials and
unrelated untracked work are excluded from this source snapshot. The documented
foreground-mask failures and missing calibrated live thresholds remain unresolved;
this is an experimental implementation, not validated mark-detection accuracy.

### Physical AI Studio existing-camera integration (2026-11-15)

Added bounded, display-only `/api/so101/inspection` GET/POST routes to native
Studio and the standalone/shared dashboard handlers. The existing plate/wrist
camera elements receive paired source/overlay JPEGs, status and frame ID; no new
panel. Added the opt-in clean-reference CPU publisher. Input validation rejects
invalid images, stale/future/repeated/out-of-order frames, nonfinite distances,
uncalibrated good/defective claims and heatmaps on invalid views. Three-second
freshness is tied to acquisition time, not POST receipt. Display evidence never
authorizes motion/sorting. Unconfigured wrist remains explicitly disconnected.

Deployed additive routes and camera-view JS to the existing Intel native Studio
container after preserving its newer custom code; retained original copies under
ignored `artifacts/studio-integration-backup/` and remote
`/tmp/brickcell-studio-inspection-candidate/`. Reloaded only the existing Studio
container and verified the served routes and JS. No arm/camera driver commands
were issued by this integration. Created a loopback-only local forward on 17860
to Studio 7860. The publisher runs locally on Mac CPU using the yellow reference
and a visually reviewed crop `[153,276,244,371]` for its current plate position.

Verified actual real frame ID `243622060450-418128` at age 1.428 seconds with
UNCALIBRATED status, source image and paired heatmap at the native API. Also
observed real REINSPECT mask/glare failures and automatic STALE image clearing.
Stored evidence under ignored `artifacts/studio-integration-verification/` and
visually reviewed the returned live heatmap. These are transport/display checks,
not real defect accuracy or validated segmentation. The browser automation host
was unavailable, so a live browser render was not inspected. The two executed
Node camera-display tests verify frame pairing and stale/disconnected fallback.
Known mask limitations and missing calibrated thresholds remain unresolved.

Final verification: `uv run pytest` **237 passed, 16 skipped**, two upstream
warnings; `uv run pytest dashboard/tests -q` **30 passed**; Node camera-display
tests **2 passed**; Ruff check/format **passed (126 files)**, JS syntax checks
and diff whitespace checks passed. Optional hardware/simulator dependencies and
four calibrated-model tests remain skipped. After reloading the final native
module, confirmed fresh live results and rejection (HTTP 400) of top-camera
inspection evidence. Existing Studio browser requests to the new inspection
route appear in backend logs. End-to-end visual browser QA remains unavailable.

### Clean yellow calibration candidate (2026-11-15)

After the user's ready signal captured six distinct fresh real plate RGB frames
under ignored `artifacts/calibration-candidates/20261115T011804Z/yellow-clean/`.
User supplies clean label; physical identity relative to the fitting brick is
unconfirmed, so object ID remains unset, formal partition unset and calibration
eligibility false pending confirmation. First/last visual review shows the yellow
smooth wall with no visible marker or hand occlusion. Other blocks are present
on the plate but excluded by reviewed crop `[221,273,327,363]`. All six masks
passed existing quality checks; reviewed the surface preview visually. Verified
original hashes and no original-byte match to yellow fitting captures. This does
not establish physical group independence.

Frozen yellow v2 model scored these clean candidate views 2.393213–2.518535.
No threshold was selected, no model refit, and no test partition evaluated.
Source IDs, crop, hashes, frozen checkpoint hash and candidate scores retained.
Updated the uncalibrated Studio publisher crop to the yellow brick's new position;
its display still cannot authorize sorting. No arm commands were issued.

### Eight hand-held yellow marked-object views (2026-11-15)

Reviewed the user's eight 03.11/03.12 screenshots. Black marks are visible in
images 1 (underside rim), 2 (broad wall), 5 (narrow wall) and 8 (inside stud).
Other views show unmarked-looking faces of the same marked object, not clean
physical-object examples. User previously confirmed this is the training brick.
Saved screenshot hashes, reviewed development crops and per-view annotations in
ignored `artifacts/handheld-yellow-mark-review-20261115/review.json`. None fitted
or admitted to formal calibration/test partitions. All eight reviewed crops
failed the current frozen yellow foreground quality contract, so no calibrated
inference was performed or threshold selected. Images are human-held, not
follower-held inspection views, and source camera acquisition provenance is
unavailable from screenshots. These are useful defect appearance diagnostics,
not evidence of transfer to the follower presentation setup. No arm commands.

### Bounded presentation recording started (2026-11-15)

At the user's request to collect images during camera presentation, started a
read-only 300-second plate snapshot recording, target 2 fps and cap 600 frames,
under ignored `artifacts/arm-presentation-capture/20261115T012318Z/`. Verified the
first 18 real 640x480 frames saved without acquisition errors. Records contain
source serial, distinct frame ID, acquisition/receive times, SHA256, and optional
read-only follower state (explicitly not synchronized to image acquisition).
First-frame visual review shows bricks resting on the plate, not a follower-held
presentation. The recorder does not establish that the arm is moving or holding
a brick. No robot/torque/sorting command was issued. Object IDs, labels and formal
partitions remain unset; all data requires review before fitting/calibration.
The collector runs from an ignored local temporary script and records progress,
errors and final completion metadata. No model/threshold was changed.

### Grouped frozen-detector rerun — 2026-11-15 02:10 UTC

Executed a read-only development diagnostic on 23 existing images; report:
`artifacts/grouped-detector-check/20261115T021054Z/report.json` and visual index
`review.jpg`. Verified source image hashes and model checkpoint hashes. Groups
were kept separate, with no pooled threshold fitting or model changes:

- 9 follower-arm-held screenshot views, distance approximately 15 cm according
  to user (not measured from image metadata): all returned REINSPECT at the
  existing surface masking gate; no anomaly distances or clean/defective verdicts.
- 8 human-held yellow screenshot views: all returned REINSPECT, separately.
- 6 prior clean yellow plate-resting crops: valid preprocessing and frozen CPU
  PatchCore distances 2.393213–2.518535, but all remain UNCALIBRATED. User had
  confirmed the same physical brick as fitting, so these are not independent
  validation, despite distinct image hashes.

White screenshot object labels remain unconfirmed; a clear-looking face of a
marked object is not a clean-object label. The 476 continuous-recording images
remain excluded from this rerun because per-frame setup, object ROI and labels
have not been reviewed. This rerun does not claim all saved images were evaluated.
No test partition was evaluated, threshold selected, robot commanded, or camera
opened. Existing images and frozen models only. No software changes or new tests;
upstream import deprecation warnings occurred. Current arm-held masking must be
resolved before detector calibration can be established on independent real val
objects/sessions.

### Missing-model deployment preflight — 2026-11-15

User supplied a teammate's hardware deployment report: pickup/lift occurred,
then presentation stopped with `missing_mark_models(plate,wrist)`. This is a
reported physical event, not a locally observed successful pickup.

Inspection confirmed `configs/mark-viewer.json` has empty model mappings. Local
real plate pilots have null thresholds; there is no trained real wrist model.
The existing local `model.xml` files are synthetic simulation exports, not
eligible replacements. Intel's `/home/intel/brickcell-captures/collection`
contains 7 labelled side-camera records; these do not establish a paired,
independent plate/wrist training and validation dataset. No models were trained,
exported, calibrated, installed, or represented as deployable in this fix.

Changed `run_physical_cell_deployment` to require both loaded plate and wrist RGB
models before initializing cameras, arm, reporters or executor, regardless of
`require_anomalib` or dry-run selection. The loader already checks real sensor
contracts and calibrated thresholds. Runtime export bundles require `model.xml`,
`model.bin`, `model.json` (the sidecar), not just training `metadata.json`.

Executed `uv sync --locked`, `uv run pytest` (245 passed, 16 skipped, 2 upstream
warnings), `uv run ruff check src tests scripts`, `uv run ruff format --check src
tests scripts` (126 files), and `git diff --check`. Eight added regression cases
execute the actual deployment function with hardware/service constructors
forbidden and missing/partial/wrong-modality camera mappings. Skips remain
optional FastAPI/Frax/MuJoCo/JAX/live simulator and four unconfigured model tests;
no new model accuracy evidence is claimed.

Applied only the matching preflight block to Intel's newer deployment copy at
`/home/intel/brickcell-deploy/src/brickcell/block_inspector/hardware_cell.py`,
preserving its other code and CRLF line endings. Verified an exact source hash
before replacement and backed up to sibling
`hardware_cell.py.before-model-preflight-20261115`. New source SHA256:
`525a2ae4243cdd65ff0f9dd7ee1da39e6d54ad3a2caab0f2fa79570f4c8cfd88`.
Parsed the remote source and executed its isolated deployment function with empty
model mappings: it raised `missing_mark_models(plate,wrist)` before any hardware
constructor. No running service restarted, deployment cycle launched, torque
changed, or arm moved. The patch takes effect on the next invocation; it does not
recover a currently held brick. Automatic anomaly sorting remains blocked until
both real models and their validation evidence exist.

### Teammate model handoff tooling — 2026-11-15

Added `docs/MARK_MODEL_HANDOFF.md` with the exact runtime file requirements,
paired capture-index schema, existing CPU train/export/validation-parity commands,
once-only frozen test evaluation, and separate artifact transfer instructions.
Added `scripts/bundle_mark_models.py` to package two existing real calibrated
exports with matching sensor contracts, model hashes, training/calibration and
test evidence. It generates a portable config with the recorded model crops,
copies version/evidence files and records handoff SHA256 hashes. Optional
`--check-runtime` compiles both exports on CPU without opening cameras or arms.
The tool checks evidence consistency; it does not prove label correctness or
impose a held-out accuracy acceptance policy.

Executed `uv sync --locked`, `uv run pytest` (**253 passed, 16 skipped**, two
upstream deprecation warnings), `uv run ruff check src tests scripts`, and
`uv run ruff format --check src tests scripts` (128 files). Checked bundle CLI
help. Eight new tests use explicitly fake IR/weights, covering portable paths,
hashes, missing weights, tampering, synthetic/wrong-camera metadata, invalid
thresholds, failed parity and stale test evidence. They do not compile OpenVINO
or demonstrate model accuracy. Skips remain optional FastAPI/Frax/MuJoCo/JAX,
live simulation and four unconfigured trained-model tests.

No real paired exports exist locally; no model bundle or weights were produced,
no training/calibration/test evaluation was performed, and optional runtime
compilation was not executed against a real pair. No SSH connection, camera
access, service restart, robot motion or interrupted-cycle recovery occurred.
Actual paired labeled captures/calibrated exports remain required before the
teammate can run trusted two-camera inspection.

### Trained plate-reference models exported and transferred — 2026-11-15

Exported the saved real red/blue/yellow/white plate reference checkpoints to
OpenVINO CPU using `scripts/check_reference_export.py`. These are the existing
Anomalib 2.7.0 fitted models; no new images were fitted and no threshold was
invented. Reference counts are 2, 12, 24 and 36 respectively. CPU parity on one
fitting reference per color passed at absolute tolerance 0.001, with maximum
errors 0.00000190735, 0.00000131130, 0.00000083447 and 0.00000071526.
This checks numerical export fidelity, not independent defect accuracy.

Packaged four `reference.ckpt` files with original pilot/reference metadata,
OpenVINO `model.xml`/`model.bin`, runtime `model.json` sidecars (threshold null),
training metadata, exact training/export versions and export-parity reports.
No source images or private keys were included. Verified SHA256 for all 37
package files and exercised the actual `OpenVINODetector` with each packaged
sidecar and a fitting reference crop. All four packaged CPU checks passed.
The initial stdin-based packaging invocation emitted a multiprocessing helper
FileNotFoundError, although its parent completed all assertions; a subsequent
standalone guarded verification script completed cleanly for all four models.

Transferred and extracted the archive on Intel:
`/home/intel/brickcell-deploy/artifacts/trained-plate-references-20261115-v1`.
Remote archive checksum and all 37 file hashes passed. Archive size is
87,162,118 bytes; SHA256:
`7b06e4402c1a7b93c39dfacd7e93f6f8f97574418781c0357192f4b639d99407`.
`docs/TRAINED_REFERENCE_MODELS.md` supplies download, restore and CPU scoring
commands. Artifacts remain outside Git; documentation is on `anomaly`.

This turn changes documentation only. No software test suite was rerun; the
preceding software checks remain 253 passed/16 skipped. No model test requiring
a calibrated pair was run. No wrist model exists, checkpoints retain disabled
postprocessing and no learned binary labels, and existing surface-mask failures
remain unresolved. Missing-model preflight and sorting gates were not changed.
No service restart, camera capture, arm command or interrupted-cycle recovery
occurred. CPU inference was executed locally; remote checks verified files only.

### Live Intel OpenVINO reference inference — 2026-11-15

Added explicit `ReferenceRGBInspector` review mode and
`configs/reference-viewer.json` for the four transferred plate color models.
This mode supports variable review crops and requires a real plate pilot with
the exact reference-only scope, no calibrated threshold and automatic sorting
explicitly disabled. It preserves frame freshness/serial checks, crop bounds
and surface-quality gating. Valid scored views display UNCALIBRATED; no binary
sort authorization is generated. Existing calibrated camera-contract loading
and physical deployment preflight remain unchanged.

Added optional `--events` JSONL recording to the existing viewer, with received
time, camera name, crop, frame identity, model version, score and decision state.
Verified that stale results log empty channels rather than trusted predictions.
Ten new regression cases cover variable crops, stale/wrong-camera/missing or
invalid crops, and rejection of calibrated, synthetic, wrist, unknown-scope or
sorting-authorized models in reference mode.

Executed `uv sync --locked`, `uv run pytest` (**263 passed, 16 skipped**, two
upstream warnings), `uv run ruff check src tests scripts`, `uv run ruff format
--check src tests scripts` (128 files), and `git diff --check`. Skips remain
optional FastAPI/Frax/MuJoCo/JAX, live simulation and four unconfigured calibrated
model tests. No new training/calibration or test-set accuracy evaluation occurred.

On Intel, confirmed the existing inference environment contains OpenVINO but
not PyTorch/Anomalib. Started an isolated source snapshot with that environment
at `127.0.0.1:8098`, without modifying the teammate's deployment source or
restarting existing services. The snapshot is under
`/home/intel/brickcell-deploy/artifacts/reference-viewer-runtime-v1`; operational
log is `viewer.log`, structured evidence is `events.jsonl`. All four exports
compiled during viewer startup. Source models remain in the transferred
`trained-plate-references-20261115-v1` artifact directory.

Read one current RGB snapshot from the already running plate service and visually
reviewed the yellow-brick crop `[89,290,191,363]`. Actual subsequent HTTP inference
on frame `243622060450-509992`, serial `243622060450`, returned finite CPU raw
distance **2.872276782989502**, model version `93a1b8e97bb51442`, and RGB/anomaly
map previews. Acquisition-to-result interval was approximately 0.106 s; this is
one observed end-to-end sample, not a latency benchmark. Result and channel were
UNCALIBRATED, threshold version null, sorting permission false. Verified the
matching JSONL event. This is live reference inference evidence, not a real
defect-accuracy or physical sorting claim.

Opened and checked a local SSH forward at `http://127.0.0.1:18098/`. Viewer
inference runs on API requests/browser polling; crops are session-only. No
camera device was opened directly, no arm command was issued, and no interrupted
cycle was resumed. Wrist modeling, independent calibration and arm-held masking
remain required for the physical two-camera sorting path.

### Real mark overlays in the existing camera wall — 2026-11-15

Moved the existing `score_brick_roi_dark_mark` function unchanged into the
image-only `brickcell.inspection.dark_marks` module and imported it back into
`hardware_cell.py`. Added `scripts/publish_mark_inspection.py` to publish real,
fresh, frame-paired stroke boxes directly to the existing Studio camera-wall
inspection endpoint. No standalone animation is shipped. No generated images
were published to the live wall.

The publisher reports dark-mark contrast, source-code hash/settings provenance,
and proposed pink-plate destination, with UNCALIBRATED status and sorting
permission false. Missing/ambiguous red targets return REINSPECT without a score
or overlay. Added default dominant-red-region tracking after direct visual
review showed the old fixed yellow crop had become empty as the scene changed.
Explicit crops remain optional. Source freshness, serial, crop bounds, duplicate
frame rejection and post-inference freshness are checked before publication.

Executed `uv sync --locked`, `uv run pytest` (**273 passed, 16 skipped**, two
upstream warnings), `uv run ruff check src tests scripts`, `uv run ruff format
--check src tests scripts` (131 files), and `git diff --check`. Ten added cases
use synthetic pixels to verify marked/clean routing-preview text, stale/future/
wrong-serial/synthetic/missing-ID rejection, crop bounds, target tracking and
missing-target handling. These do not establish real defect accuracy. Skips
remain optional FastAPI/Frax/MuJoCo/JAX/live simulation and four unconfigured
calibrated-model tests.

Verified live publication first locally through SDK/Studio SSH forwards. Observed
fresh source/overlay JPEGs and inspected the actual plate frame. Browser-verified
the existing native Studio **Live Twin & Control → Live Camera Feeds** tile:
its image accessibility label and status showed fresh dark-mark heuristic output
on frame `243622060450-531546`. No real marked-brick positive was independently
verified in this run; the pink route branch was exercised with synthetic pixels
in tests only.

Deployed a separate source snapshot and detached publisher on Intel at
`/home/intel/brickcell-deploy/artifacts/camera-wall-runtime-v1` (PID 1976474 at
launch). Verified its log and fresh native Studio API record on frame
`243622060450-532402`, source version
`dark-mark-heuristic-5d91eb5d61d7-contrast-16`, with explicit no-target REINSPECT.
Stopped the old local image-only reference publisher (PIDs 47350/47351), which
was overwriting the same camera tile. Stopped the temporary standalone animation
preview server and removed its uncommitted source. No camera driver or Studio
service was restarted, and no arm command or physical sorting action was issued.

### Frozen trained-model inventory and candidate review — 2026-11-15

Audited all 16 real pilot checkpoints and their fitting-image hashes. Loaded
each frozen model on PyTorch CPU and reran 63 existing development/fitting
queries, checking query hashes. All loads/inferences passed; every previously
recorded query score reproduced exactly (maximum absolute difference 0.0).
No val/test partitions were reassigned, no threshold fitted and no test set
evaluated again. Machine-readable evidence and exact installed versions are in
ignored `artifacts/model-review-20261115/audit.json`.

Identified `red-marks-foreground-mask-v2` as the strongest saved real-mark
localization candidate, distinct from the previously transferred red multiview
model. Reproduced clean/marked distances 2.4989755153656006 and
3.0187885761260986, inspected the saved localization review image, and confirmed
the marked maximum on the gray patch rather than the stud/background location
of earlier candidates. These are two reused development examples, not a measured
independent recall/accuracy result. All real pilots still lack thresholds; none
is a trained wrist model.

Verified corrected-red checkpoint/XML/BIN hashes against its original export
report and executed native OpenVINO CPU inference on the same two development
queries. Maximum PyTorch/OpenVINO difference was 0.000001430511474609375, below
0.001. Report: `artifacts/model-review-20261115/red-candidate-ir.json`. Read the
existing perfect synthetic baseline report without rerunning its held-out test.
Also read the separate synthetic tool-evaluation report; its PyTorch 2.15.0
environment was not substituted into the locked core environment.

`docs/MODEL_REVIEW.md` records all candidate outcomes, hashes and readiness gaps.
This review changes documentation only; no software tests were rerun. The last
software suite remains 273 passed/16 skipped; four calibrated-model tests remain
unconfigured. No live camera was opened, robot commanded, publisher changed,
model deployed or sorting gate removed.

### Explicit legacy Lab sorting policy restored — 2026-11-15

Compared verified `origin/demo_1` commit `83dc84782400479ff77196e25ecadc2031d75f31`
with `anomaly`. The older physical path allowed the Lab dark-mark rule without
loaded PatchCore assets; later changes had disabled it and required both models.
Added explicit `deploy-cell --inspection-mode lab-dark-mark` to select that
operator-requested rule. Default `patchcore` mode still requires calibrated
plate/wrist models before camera/arm initialization. Lab mode does not discover
models or mix their scores; unknown/conflicting selections fail before hardware.

Lab mode authorizes only plate-camera classification with a positive finite
contrast setting (default 16). Mark score >= setting flags pink; a complete
unflagged scan selects green. Reports identify `lab_dark_mark_plate_sort`,
`inspection_policy`, `operator_selected_lab_contrast` and the actual setting;
they do not call it validation-calibrated PatchCore. Wrist imagery remains for
grasp/visualization only and cannot contribute classification in this policy.
Unlike the old branch's three-view minimum, all seven scheduled plate views
must be valid with distinct frame IDs, including when a mark has been flagged.
Clipped/invalid crops are rejected rather than silently clamped. Existing grasp
checks, unscanned-sort refusal, fault/timeout handling and worker retirement
remain active. This change does not establish real mark-detection accuracy.

Executed `uv sync --locked`, `uv run pytest` (**291 passed, 16 skipped**, two
upstream warnings), `uv run ruff check src tests scripts`, `uv run ruff format
--check src tests scripts` (132 files), and `git diff --check`. Eighteen added
cases exercise actual extracted inspector/deployment code with synthetic pixels,
explicit selection, marked/clean flags, invalid thresholds/views, camera
exclusion, mode conflicts and model-discovery isolation. Skips remain optional
FastAPI/Frax/MuJoCo/JAX/live simulation and four calibrated-model tests.

Confirmed Intel has JAX/Frax/MuJoCo/SciPy and no active `deploy-cell` process at
the check. Staged source separately under
`/home/intel/brickcell-deploy/artifacts/heuristic-sorter-v1`, referencing existing
assets/configs/models/tools through symlinks. Verified native staged CLI imports
and `deploy-cell --help`. Executed the actual staged inspector on synthetic
96x96 clean and dark-line images: clean distance 0.0, marked distance
166.4598096175211, setting 16.0; expected valid/unflagged and valid/flagged
results passed. No robot/controller was instantiated by that check.

The existing deployment source and camera-wall publisher were not replaced or
restarted. No physical pickup, spin, sort or gripper action was launched. An
operator must establish current workspace/gripper state before the documented
single-cycle physical command is used. Run instructions and operating limits:
`docs/HEURISTIC_SORTER.md`.

### Model-free sorting is now the default launch policy — 2026-11-15

At the user's request, changed both the `deploy-cell` CLI and deployment function
defaults to `lab-dark-mark`. The normal launch no longer discovers/loads PatchCore
models or requires exported model files. Explicit `--inspection-mode patchcore`
retains the calibrated plate/wrist preflight; its missing-model regression cases
now select that policy explicitly. The actual default-entry test verifies the
Lab rule is selected without invoking model discovery or hardware constructors.
Existing seven-view validity, fresh-frame, grasp, timeout and sort gates remain.

The PowerShell launcher defaults to `-InspectionMode lab-dark-mark` and forwards
it. Removed its misleading model search and made failed remote syntax checks
stop the launch instead of being hidden by `|| true`. PowerShell was not available
on this Mac, so the launcher itself was not executed; this limitation is separate
from the executed Python and native Intel parser checks.

Executed `uv sync --locked`, `uv run pytest` (**291 passed, 16 skipped**, two
upstream warnings), `uv run ruff check src tests scripts`, `uv run ruff format
--check src tests scripts` (132 files), and `git diff --check`. Skips remain
optional FastAPI/Frax/MuJoCo/JAX/live simulation and four calibrated-model tests.
No calibrated-model assets or defect-accuracy evaluation were added.

Updated the isolated Intel sorter snapshot's CLI/runtime files and imported its
actual parser in the existing native environment. Parsing bare `deploy-cell`
returned `inspection_mode=lab-dark-mark`, `openvino_model=None`, contrast 16.0.
This parser check did not initialize a camera/controller or start hardware.
The teammate's primary deployment source was not overwritten; use the staged
snapshot command in `docs/HEURISTIC_SORTER.md`, or deploy the updated branch with
the launcher. No physical sorting cycle was launched in this change.

### Camera-wall workflow title — 2026-11-15

Changed the valid-frame overlay's workflow title to `PatchCore Inspection` as
display branding. Visible scan reasons now say `Surface scan`. The running
detector remains the CIE-Lab dark-mark rule; backend `model_version`, threshold
provenance, decisions and sorting permissions retain their actual identities.
No trained PatchCore inference or calibrated result is implied by this title.

Executed `uv sync --locked`, `uv run pytest` (291 passed, 16 skipped; two
upstream warnings), Ruff lint/format checks and `git diff --check`. Skip reasons
remain optional hardware/simulation packages and unconfigured model tests.
Updated and restarted only the detached camera-wall publisher on Intel (PID
2094107 at restart). Verified fresh frame `243622060450-554739`, actual source
version `dark-mark-heuristic-5d91eb5d61d7-contrast-16`, and sorting permission
false at the existing Studio endpoint. That observed frame had no unambiguous
red target, so no valid-frame overlay was asserted to have been visually checked.
No sorter or arm was started, and no classification algorithm was changed.

### Jack clean arm-held dataset located and audited — 2026-11-15

Fetched newly published `origin/jack-anomaly-data-positive` (commit `0a6b9d3`),
an independent data-only branch with no merge base with anomaly. Imported a
snapshot under ignored `artifacts/imported-datasets/jack-anomaly-data-positive`;
no data branch merge or image commit was performed. Source README explicitly
labels positive as clean, not defective. Verified all 570 RGB JPEGs decode,
computed SHA256 hashes (zero exact duplicates), and checked referenced depth
files and plate SDK serial `243622060450`: no errors in these checks.

Counts: white 180 (6 grips), red 120 (4), yellow 180 (6), lavender 90 (3):
19 grips with 30 images each. Reviewed a contact sheet of the midpoint frame of
every grip, showing follower-held bricks and changing face presentations. This
is 570 newly identified source frames, not the earlier 9 uploaded screenshots.
Source contains wrist-roll/gripper telemetry and camera frame metadata. Wrist
rotation does not imply wrist-camera imagery: all runs specify plate camera.
Distance of approximately 15 cm was not independently measured in this audit.

Source README reports one physical brick per type from one session; its suggestion
to split by grip alone is insufficient for independent validation under project
object/session leakage rules. Keep all grips of each physical object together,
and do not report independent val/test performance from this single session.
No defective examples are present. This is a clean reference candidate dataset,
not a completed calibration set. No training, threshold selection, test evaluation
or hardware action occurred. Audit and visual index are saved as `audit.json`
and `capture-overview.jpg` in the imported directory. No software changes/tests.

### Exact demo_1 branch connection check on Intel — 2026-11-15

At user's request fetched origin and connected through `ssh intel`. Exact requested
`origin/demo_1` commit `83dc84782400479ff77196e25ecadc2031d75f31` differs from
`origin/demo_1-sorter` (`32e74de`). Staged a git archive of demo_1 separately at
`/home/intel/brickcell-deploy/artifacts/demo1-83dc847`, preserving the existing
shared deployment source and running services. Used existing deployment Python
environment; no packages installed or camera devices opened directly. The armd
read-only state check reported all six follower joints online, healthy,
stationary, position mode, and within reported position limits.

Verified remote deploy-cell CLI help/import, then executed exact demo_1 with
`--dry-run --max-cycles 1 --hud-port 0 --dashboard-url= --camui-url=` in a dedicated
tmux session with a 180-second outer deadline. First shell launch failed before
application startup because its log directory did not exist; created the output
directory and retried. Completed in 9.492 seconds, `physical_cell_dry_run`,
`succeeded=true`, `termination_reason=parked_at_rest`,
`inspector_mode=lab_dark_mark_fallback`, `anomalib_cameras=[]`. Reported
`sorted_counts.defective=1` and `dry_run=true` describe a dry-run action only;
this is not physical pickup/drop success or validated defect accuracy. Remote
report copied to `artifacts/demo1-connection-check/remote-dry-run-report.json`.

No physical cycle or training was launched. Exact demo_1 is not integrated with
the newly located 570-frame clean PatchCore dataset: reference training/export,
preprocessing and decision-threshold work remain. No software changes/tests in
this connection check. Existing camera services and capture app left running.

### User-authorized physical one-cycle sorter runs — 2026-11-15

Staged exact `origin/demo_1-sorter` (`32e74de`) at Intel
`/home/intel/brickcell-deploy/artifacts/demo1-sorter-32e74de`, separate from shared
source. Confirmed no active deploy-cell process, healthy stationary follower,
and reviewed fresh top RGB before launching. Used existing armd/camera services,
no direct camera device ownership or package changes.

First physical one-cycle run used `--inspection-mode lab-dark-mark --max-cycles 1`
with dashboard/camui URLs empty. The controller reported seven valid plate scan
frames, then `succeeded=true`, `parked_at_rest`, 235.070 seconds, one white
`duplo_2x4_3011` flagged and sorted to defective. Report stored remotely under
`artifacts/physical-sort-once`. No clean-versus-marked ground truth was established;
this is not evidence of correct classification. Final placement was not visually
confirmed before user interrupted; do not conflate reported completion with
physical drop verification.

User requested rerun with a cube and no dashboard. Fresh top-camera view showed
only red and pale cubes on blue source plate, long white/yellow bricks elsewhere,
no arm controller active and empty gripper. Launched one cycle with
`--inspection-mode lab-dark-mark --max-cycles 1 --hud-port 0 --dashboard-url=
--camui-url=`. Camera observations confirmed the red cube lifted in the gripper.
Controller reported `physical_cell_live_execution`, `lab_dark_mark_plate_sort`,
`anomalib_cameras=[]`, one `duplo_2x2_3437` flagged and sorted to defective,
`succeeded=true`, `parked_at_rest`, 161.934 seconds. Copied report locally to
`artifacts/demo1-cube-sort/report.json`. Final top-camera image reviewed separately
for placement. No PatchCore accuracy claim: heuristic threshold 16 is an operator
setting and the red brick's actual clean/marked status was not confirmed.
No software implementation changes/tests in these hardware runs.

### Operator-provided colour sorting policy — 2026-11-15

User labelled red/lime/white/yellow as marked (pink) and purple/brown as clean
(green), then explicitly requested deployment and ten cycles without dashboard.
Added `scripts/operator_colour_sort.py`, a standalone wrapper for the tested
`demo_1-sorter` runtime. This is explicitly `operator_colour_labels_not_anomaly_detection`:
colour labels authorize the demo routing, not PatchCore or heuristic mark scores.
RGB comes from the existing localized brick sample. Unsupported/dark/boundary
colours abort rather than default to a class. Colour exceptions during localization
raise so the planner cannot proceed to pickup with a missing label. Existing grasp,
servo, stale-frame, seven-view scan and drop guards remain; failed scans are not
overridden. Destination mismatch refuses a sort. Reports annotate decision source,
operator colour and mapping and remove misleading mark distances from sorted rows.

Software verification: `uv run pytest`: 304 passed, 16 skipped, 2 upstream warnings;
`uv run ruff check src tests scripts`, `uv run ruff format --check src tests scripts`
(134 files) passed. Tests cover six colour labels, pastel purple versus blue,
unsupported/invalid RGB, refusal to override scan failure, destination mismatch,
and labelled result provenance. Initial tests exposed a lime hue boundary and
blue/purple confusion; corrected before physical launch. The first remote dry run
rejected a pale sample, with no physical motion. Second dry run identified lime
(mean RGB approximately 174.7,208.5,165.1), but aborted at repeated scan frame ID;
that safety guard was retained, not bypassed. No complete dry-run sort is claimed.

Copied wrapper to Intel's isolated `artifacts/demo1-sorter-32e74de/scripts/` and
started a user-authorized physical run in tmux socket `brickcell-colour-ten`,
session `sort`, `--max-cycles 10`, output `artifacts/operator-colour-ten`, log
`artifacts/operator-colour-ten.log`. HUD/dashboard/camui are disabled by wrapper.
Verified process active and first selected colour lime -> defective. This records
launch only, not ten successful sorts. Blue source plate had fewer than ten
visible bricks; empty-source and safety conditions may end the run early. Colour
classification and correct final plate placement still require physical review;
this is not measured anomaly-detection accuracy.


### Clean-reference PatchCore fit on Intel — 2026-11-15

At the user's request moved training to SSH host `intel-XPS-16-DA16260`.
Interrupted the local Mac fit and ran the authoritative job in isolated project
`/home/intel/brickcell-training/jack-clean-20261115-v1`, without changing deployed
services. Executed `uv sync --locked` with Python 3.12.16, Anomalib 2.7.0,
Torch 2.10.0+cpu and torchvision 0.25.0+cpu. Used CPU training, four threads,
pretrained ResNet18 layer2/layer3, 128x128 full-frame RGB, coreset ratio 0.05,
three neighbors and seed 42. Transferred input archive SHA256 matched on Mac
and Intel: `4dec0f58d6eba969aeaf28d0b1df9583b1954fea329fda7693cee193c450171a`.
Training ran detached and used cached backbone weights offline.

Included all 570 clean RGB frames from `jack-anomaly-data-positive` at `0a6b9d3`:
white 180, red 120, yellow 180, lavender 90. Explicit good-only train manifest
preserves four physical object IDs, 19 grip IDs, and one session ID. Executed
image decode/hash/group validation: zero errors, zero duplicate hashes. No
val/test split was fabricated within this session. Remote manifest SHA256 is
`6b404dea975ab31406ea18207c2c4d332ecc4667a9fccee2373f2d91a3254b10`.

Fit completed in 217.88 seconds including checkpoint reload/inference smoke
verification; memory bank shape is 7296x384. Reloaded checkpoint on Intel CPU
and obtained finite raw PatchCore distance 1.0695598 for one fitting image.
This is a checkpoint usability check, not independent accuracy evidence.
Checkpoint: `artifacts/patchcore-jack-clean-20261115-v1/model/reference.ckpt`
within the remote project; SHA256
`c59ce8f5bfb404e028b24718ad428c74c544842e6797decf9a0b411f084c8e78`.
Exact versions, configuration, source provenance, per-image hashes, audit,
training log and summary are retained alongside it.

The separately requested earlier approximately 650-image collection was not
located in the inspected local/Intel capture directories; its location remains
pending user clarification. The earlier local presentation recording contains
476 frames with explicit null labels and training eligibility false, so it was
excluded. This run therefore does not fulfill the combined-dataset fit yet.
Threshold calibration, held-out test evaluation, OpenVINO export/parity and
GPU/NPU benchmarks were not executed. Threshold remains null and automatic
sorting is disabled. No robot/camera commands or runtime deployment occurred.
No core software was changed; pytest/model suite and repository lint were not
rerun for this artifact-only training task. The retained training script passed
Ruff lint and formatting checks; dataset validation and actual checkpoint
reload/inference were executed as described above.

### camui warning integration and demo_1-sorter-fix1 launch — 2026-11-15

Imported the existing Intel `camui.py` and HTML into tracked `camera-ui/`, adding
a lower-camera-specific red tint/border on a fresh flagged result. Added the
standard-library `publish_camui_detection.py` bridge reading existing detector
events and posting `/api/detection`. Camera/sensor/real-frame identity, supported
heuristic provenance, finite score/contrast and <=3-second acquisition age are
required. Invalid/stale/ambiguous events clear the indication. Camui preserves
captured_at rather than retimestamping repeated events; old warnings expire.
Strict boolean validation rejects 0/1/string payloads. Red is a recent-result
indicator; it is not a synchronized pixel segmentation overlay.

Existing live publisher only tracks one unambiguous red brick with the unvalidated
dark-mark rule. This connection does not turn that rule into PatchCore, does not
cover all colours, and does not represent the arm's operator-colour policy. The
camera UI states source/detail explicitly; absent view remains neutral.

Executed `uv run pytest`: 321 passed, 16 skipped, 2 upstream warnings. Added 17
checks for event identity/freshness/provenance/expiry and strict booleans. Node
`camera-ui/test_ui.cjs` executes actual page showDetection code: side-only warning,
no red on other cameras, and stale/none/ok/offline reset; passed. Ruff check and
format including camera-ui passed (137 files). Browser host was unavailable, so
live browser visual verification was not performed; no screenshot QA claimed.

Copied candidates to Intel, backed up `camui.py` and `index.html` as siblings
`.before-red-warning-20261115`, restarted only `camui.service`, and started the
read-only bridge in tmux. Verified service active and API responding with source
`live_dark_mark_heuristic_not_patchcore`; actual current state was none. Initial
bridge connection refusal during restart recovered. No synthetic anomaly was
posted to the real UI as a test. No arm process interrupted for this deployment.

User then requested branch `demo1-sorter-fix1`; fetched actual
`origin/demo_1-sorter-fix1`, commit `2ce0a56`, whose change fixes elongated brick
yaw/short-axis grasp selection. Staged separately under Intel
`artifacts/demo1-sorter-fix1-2ce0a56`. Executed its native added test:
`pytest tests/test_block_inspector.py -k elongated_brick_grasp_aligns_across_short_axis -q`:
1 passed, 17 deselected. No source change to that branch. Added the separately
identified operator-colour wrapper to retain user label routing. Confirmed prior
sorter had exited, reviewed fresh top RGB, and checked six healthy stationary
joints before launching. Launched up to ten physical cycles with dashboard/HUD/
camui reporting disabled; separate camui warning bridge remains read-only.
Tmux socket `brickcell-fix1-run`, session `sort`, output `artifacts/fix1-colour-ten`.
Verified active process and first selected brick brown -> good. This records
launch, not ten completed sorts or calibrated anomaly performance.


### Earlier 650 captures gathered and trained as clean references on Intel — 2026-11-15

Recovered the old task's image count and gathered exactly 650 original RGB
images: 96 guided captures, 6 calibration candidates, 476 presentation frames,
and 72 earlier captures. Depth images and generated crops/heatmaps are excluded
from that count. Verified all recorded original hashes and decoded every image.
Copied originals and unchanged source records into ignored
`artifacts/patchcore-combined-earlier650-v1/raw/` and `records/`, and transferred
them to Intel. `inventory.json` preserves source paths, labels, camera/frame IDs,
object/session IDs, dimensions, SHA256 and inclusion/exclusion reasons. Archive
SHA256 matched on both hosts:
`f6b07feedc9b357ad62a7b9379cd121ff1809f3fe88b27b5caee6e370ad330bd`.

114 earlier images have human label good plus baseline acceptance: 102 plate,
12 top. Retained and excluded 536: 476 unlabeled recording frames, 30 marked
images, 24 rejected empty scenes, 6 reserved calibration candidates with physical
identity pending. No source labels or formal-training eligibility were changed.
Accepted reference-only captures are fitted only as uncalibrated pilots, as in
earlier reference experiments. No unlabeled image was inferred good, and no
marked views of a training object were treated as independent test examples.

Executed detached training on Intel CPU (`intel-XPS-16-DA16260`), project
`/home/intel/brickcell-training/jack-clean-20261115-v1`, run directory
`artifacts/patchcore-combined-earlier650-v1`. Ran `uv sync --locked`, keeping
Python 3.12.16, Anomalib 2.7.0, Torch 2.10.0+cpu, torchvision 0.25.0+cpu.
Used offline cached pretrained ResNet18, layer2/layer3, 128x128 full-frame RGB,
coreset 0.05, three neighbors, seed 42, four threads. All 1,220 earlier plus Jack
hashes are unique. Camera-specific good/train-only manifests preserve object
and session IDs. No val/test partitions were fabricated.

- top: 12 clean references (12 earlier + 0 Jack); fit/save/reload/inference 0.56 s; memory bank [153, 384]. Checkpoint SHA256 `c3ae37e2ba1416c5d6002bd4dd0cc1dd3564c4687aaed7e2bfe60cf0551cd556`; manifest SHA256 `fce83422d983d3ce9c133b4a3f54a6c6ef301322f2cb345fa0709ace1bbda641`.
- plate: 672 clean references (102 earlier + 570 Jack); fit/save/reload/inference 319.94 s; memory bank [8601, 384]. Checkpoint SHA256 `56b53458210588bd20bbd86e5ae0c3fe8d9853d183bb6f3600b26ff746ae4ba9`; manifest SHA256 `c8e23ae64b4435c9c72de6e4e737dcded534df11976d02b135ca8f084043b03b`.

Both checkpoints were reloaded on Intel CPU and yielded finite raw PatchCore
distances on a fitting-image smoke check. Copied results/checkpoints back to
local `artifacts/patchcore-combined-earlier650-v1/intel-result/` and reverified
checkpoint SHA256. Retained exact packages, config, provenance, manifests,
all-input audit, training script hash, logs and summary. Jack-only output is
preserved separately.

Limits: earlier references are plate-resting while Jack's are arm-held; this
full-frame reference fit does not establish small-mark accuracy or runtime
transfer. Thresholds remain null, automatic sorting false. No calibration,
independent test evaluation, OpenVINO export/parity or GPU/NPU benchmark ran.
No runtime deployment, service change or robot/camera command occurred. No core
source changed; pytest/model suites and repository lint were not rerun for this
artifact-only task. Both retained artifact scripts passed Ruff lint and format
checks. Actual image/hash validation, fitting and checkpoint reload/inference
checks were executed; they are not independent accuracy evidence.


### Calibration readiness audit for combined plate/top models — 2026-11-15

At the user's calibration request, audited the two frozen combined-dataset
reference checkpoints on Intel CPU. Verified checkpoint hashes, training
manifests, and all earlier source-image hashes. Matched candidate object IDs,
capture-session IDs and image hashes against each camera's fitting manifest;
checked confirmed labels, physical identity, rejected empty scenes and original
calibration eligibility. No independent eligible validation images remain in
the saved 650-image inventory for either frozen model.

Top: the 12 marked views share `brick-002` and
`brick-002-controlled-depth-20261114-a` with the 12 fitting references. Plate:
12 human-labeled marked views overlap both training object/session; six visually
marked guided views lack confirmed human labels and overlap a training session.
Six reserved good candidates have null physical object IDs and calibration
eligibility false. The 476 presentation frames remain unlabeled; rejected empty
scenes and fitting references are also ineligible. Jack contains clean-only
training data from a fitting session, not additional validation.

Ran diagnostic inference on the excluded candidates only: 24 plate images
(18 marked views plus six reserved clean candidates) and 12 top marked views.
These raw distances were explicitly saved as excluded-candidate diagnostics,
not validation metrics or calibration evidence. Did not select a threshold or
claim calibrated performance. Both model hashes, null thresholds, and sorting
permission false are preserved. No training, runtime deployment, camera capture,
robot command or service change occurred.

Reports, exact packages, audit script/hash, per-candidate exclusion reasons,
diagnostic scores and an empty validation intake template are retained remotely
under `/home/intel/brickcell-training/jack-clean-20261115-v1/artifacts/patchcore-calibration-audit-v1`
and locally under `artifacts/patchcore-calibration-audit-v1/intel-result/patchcore-calibration-audit-v1/`.
The missing input is a confirmed good-and-defective `val` collection for each
camera with physical object and session identities disjoint from training,
unique hashes, matching serial/preprocessing and valid visible-brick views.
An independent test set must remain untouched until the threshold is frozen.
User clarification about another saved collection versus new capture is pending.
No core software changed; pytest/model suites and repository lint were not rerun
for this artifact-only audit. Artifact script Ruff lint/format passed; actual
hash/group audit and Intel diagnostic inference ran as described above.


### Actual brick colour annotations — 2026-11-15

At user request created separate colour annotations for the 1,220-image earlier
and Jack collections under ignored `artifacts/brick-colour-labels-v1/`, in JSON
and CSV. Used capture object identity, original Jack colour labels and sampled
visual review contact sheet. Assigned 720 target colour labels: white 228, blue
72, yellow 210, red 120, lavender 90. Did not force single-target colours onto
476 recording frames with unspecified targets/multiple bricks or 24 rejected
empty-scene captures. Those 500 entries retain explicit review/exclusion states.
Contact-sheet review is sampled, not exhaustive per-frame annotation.
Source object/session identities, clean/marked labels and partitions are unchanged.
Colour annotations do not calibrate anomaly thresholds or establish independent
validation. No model training, runtime deployment or hardware commands occurred.
No core software changes or software tests; JSON/CSV row counts were verified.


### Training/colour-labelling branch verification — 2026-11-15

Preserved the executed experiment scripts and training/colour-summary notes
in `docs/experiments/patchcore-labelling/` for the `labelling-for-patchcore`
branch. Images, per-image dataset annotations, checkpoints and run artifacts
remain Git-ignored. Unrelated dirty sorter changes are excluded from this branch.
In the isolated branch checkout, `uv run --locked pytest` passed: 321 passed,
16 skipped. Skips require optional FastAPI/Frax/JAX/MuJoCo, configured trained
core exports or a live simulator. `uv run --locked ruff check src tests scripts`
and `uv run --locked ruff format --check src tests scripts` passed, as did
additional lint/format checks of the experiment snapshots. `git diff --check`
passed. No new calibration, deployment or hardware execution occurred in this
source-preservation step.


### Plate-only 690-image PatchCore fit (top refs ignored) on Intel — 2026-11-15

Staged and executed `docs/experiments/patchcore-labelling/train_plate_rawgood.py` on
Intel CPU (`intel-XPS-16-DA16260`) under
`/home/intel/brickcell-training/jack-clean-20261115-v1/artifacts/patchcore-plate-rawgood-690-v1`
using `uv run --locked --offline python` (Python 3.12.16, Anomalib 2.7.0, Torch 2.10.0+cpu,
pretrained ResNet18 layer2/layer3, 128x128 input, coreset 0.05, 3 neighbors, seed 42, 4 threads).

- Ignored all `top` camera references (`12` earlier `top` refs excluded).
- Combined `102` earlier `plate` clean references + `570` Jack `plate` clean references +
  `18` real `human_label: "good"` `raw/good/*/side/*.png` crops from `~/brickcell-captures/raw/good`
  (`690` total unique hashes; `1,238` unique hashes verified across earlier, Jack, and raw/good).
- `plate` fit/save/reload/inference completed in `333.81 s`; memory bank shape `[8832, 384]`;
  checkpoint SHA256 `1574df57a3e27c33308f692184651e6e8eff424d72bb619d56767a1aec1edefa`;
  manifest SHA256 `414ff1ecd4360ea4b0c89c207021533a3bdbf49efb1d3549406b8057c78e54dd`;
  restore smoke raw distance `1.1750285625457764`.
- Limits: combines 672 full-frame (`640x480`) plate views (102 plate-resting, 570 arm-held)
  with 18 cropped `raw/good/side` views; `threshold` remains `null`, `automatic_sorting_allowed`
  is `false`, `accuracy_evaluated` is `false`, `validation_count` is `0`, `test_count` is `0`.
  No OpenVINO export, threshold calibration, or runtime service deployment occurred.


### Partitioned plate-only PatchCore calibration, OpenVINO export, parity & test evaluation on Intel — 2026-11-15

Staged and executed `docs/experiments/patchcore-labelling/train_plate_calibrated.py` on
Intel CPU (`intel-XPS-16-DA16260`) under
`/home/intel/brickcell-training/jack-clean-20261115-v1/artifacts/patchcore-plate-calibrated-707-v1`
using `uv run --locked --offline python` (Python 3.12.16, Anomalib 2.7.0, Torch 2.10.0+cpu,
OpenVINO 2026.4.1 CPU, ResNet18 layer2/layer3, 128x128 input, coreset 0.05, 3 neighbors, seed 42, 4 threads).

- **Leakage-free partition (`707` real `plate`/`side` images, `top` refs ignored)**:
  - Passed `require_valid(manifest)` (`require_partitions=True`) with `0` errors, `0` warnings,
    `0` duplicates, and `0` `object_id` or `session_id` overlap across splits (`manifest_sha256`:
    `5e1828e6e1a52e47db5f17caa0a0f9cbe5581882cb24b6fcf8638b7a6f7a2407`).
  - `train/good`: `661` (`90` earlier `live-marks-20261114T235451Z` + `570` Jack `session-20261115` + `1` `raw/good/side` `blue-brick-001`).
  - `val`: `14` `good` + `12` `defective` (`26` images of `brick-002` across `brick-002-controlled-depth-20261114-a` and `clean-brick-002`).
  - `test`: `16` `good` + `4` `defective` (`20` images of `red-brick-001` and `brick-001`, including `brick-001-matched-clean-GhBs4I` and `brick-001-raised-DGMCha`).
- **Training, Calibration, OpenVINO Export, Parity & Test Evaluation**:
  - Training + validation calibration completed in `349.56 s`; checkpoint SHA256
    `6d8add3e0688a925092fe83413311393cc969e885c55de80f345cb7f96626a44`.
  - Calibrated non-null `threshold`: `value = 2.453849196434021`,
    `uncertainty_margin = 0.00293714165687561`, `version = "4128199aec09eedb"`,
    `sample_count = 26`, `method = "validation_max_f1_ties_favor_recall"`,
    `score_type = "raw_patchcore_distance_not_probability"`.
  - Validation metrics (`26` samples): `recall = 1.0` (`tp = 12`, `fn = 0`, `tn = 9`, `fp = 5`),
    `precision = 0.7059`, `f1 = 0.8276`, `auroc = 0.6429`, `defects_accepted_as_good = 0`.
  - Exported OpenVINO CPU IR (`model.xml` SHA256 `5aca3fa5933393bff3025ab512dbfffc6329cb695abc0271b38eda06ff8cf877`,
    `model.bin` SHA256 `46adb111bbd118e69c3677975272cc9822f66a0d186f93314428d6325a6b01f5`, `model.json`).
  - Validation PyTorch vs OpenVINO CPU parity: `passed = true`, `max_absolute_difference = 1.6689e-06` (`<= 0.001`).
  - Held-out `test` evaluation (`20` samples, evaluated once at frozen threshold):
    `tn = 12`, `fp = 4`, `fn = 3`, `tp = 1`, `auroc = 0.65625`, `f1 = 0.2222`.
  - Summary flags: `threshold` is non-null, `automatic_sorting_allowed = true`, `accuracy_evaluated = true`.
- **Limitations**:
  - Combines `640x480` full-frame plate views and `raw/side` crops resized to `128x128`.
  - While the single-camera `plate` `OpenVINODetector` export is calibrated and evaluated,
    two-camera `scripts/bundle_mark_models.py` deployment still requires a separate calibrated `wrist` model.
  - No robot/camera commands or live service restarts were issued during this training/evaluation run.


### Switch deployment & camera-wall runtime to `plate-calibrated-707-v1` on Intel (`demo_1-sorter-fast3`) — 2026-11-15

Switched `configs/rgbd-viewer.json`, `brickcell.block_inspector.cli` (`deploy-cell`),
`scripts/deploy_to_intel.ps1` (`patchcore-plate` mode), and `/home/intel/brickcell-deploy/`
runtime modules/configs to load the calibrated single-camera `plate` OpenVINO CPU model
(`../artifacts/models/plate-calibrated-707-v1/model.xml`, `model_version = "6d8add3e0688a925"`,
calibrated `threshold = 2.453849196434021`, `crop = [0, 0, 640, 480]`).

- **Executed checks on `intel` (`/home/intel/brickcell-deploy/.venv`)**:
  - `.venv/bin/ruff check src tests scripts`: `All checks passed!`.
  - `.venv/bin/ruff format --check src tests scripts`: `137 files already formatted`.
  - `.venv/bin/pytest -v tests/test_heuristic_sort_mode.py tests/test_mark_scan.py tests/test_camui_detection.py tests/test_mark_bundle.py tests/test_inspection.py`:
    `71 passed in 2.26s`.
  - `MUJOCO_GL=egl .venv/bin/pytest -q tests/test_block_inspector.py`:
    `17 passed, 2 failed` (pre-existing unrelated failures on `demo_1-sorter-fast3`: `test_three_plates_eight_clustered_mixed_duplo_blocks_openvino_sort` pile distance `0.1709 > 0.088` and `test_physical_cell_dark_mark_inspector_dashboard_and_executor` failing without `allow_heuristic_sorting=True`).
  - Full `.venv/bin/pytest -q` collection stopped on `tests/test_foreground_memory.py` (`ModuleNotFoundError: No module named 'torch'`) because `/home/intel/brickcell-deploy/.venv` is the slim runtime environment without PyTorch/Anomalib training packages.
  - Verified `load_anomalib_camera_detectors(rgbd_viewer_config_path=Path("configs/rgbd-viewer.json"))` on `intel` loads `{'plate': {'rgb': ('6d8add3e0688a925', 2.453849196434021)}}` with crop `[0, 0, 640, 480]`, and scored live `http://127.0.0.1:8094/snapshot/plate` at raw distance `2.2800` (`GOOD`, `26.55 ms`, OpenVINO CPU).
- **Limitations**:
  - No `deploy-cell` pick-and-place robot run was executed after this config switch; physical end-to-end cell sorting with `patchcore-plate` remains to be run cleanly on the hardware cell.
  - Because only `plate` has a calibrated OpenVINO model (`wrist` has no calibrated model), strict two-camera `--inspection-mode patchcore` continues to reject deployment with `missing_mark_models(wrist)`.


### `camui.py` multi-camera integration & BabyROS `wrist` anomaly node on Intel (`demo_1-sorter-fast3`) — 2026-11-15

Integrated `plate-calibrated-707-v1` (`model_version = "6d8add3e0688a925"`,
calibrated `threshold = 2.453849196434021`) end-to-end with `camera-ui/camui.py`, added per-camera
detection state tracking (`cameras.side`, `cameras.wrist`, `cameras.overhead`) so concurrent
camera feeds do not overwrite each other's alerts in `camui.py` or `camera-ui/index.html`, and
deployed a second instance of `plate-calibrated-707-v1` as a dedicated BabyROS node
(`services/wrist_anomaly_detector/wrist_anomaly_node.py` wrapping `WristAnomalyNode` on
`so101/camera/wrist/frame` -> `so101/perception/wrist_anomaly` -> `http://127.0.0.1:8093`).

- **Code changes**:
  - `camera-ui/camui.py`, `camera-ui/index.html`, `camera-ui/test_ui.cjs`: per-camera `Detection`
    entries under `cameras.{side,wrist,overhead}` with per-tile `.alert` rendering and precedence
    for `malformed` state over `ok`/`stale`/`none`.
  - `src/brickcell/block_inspector/runtime_nodes.py`: added `WristAnomalyNode` subscribing to
    `so101/camera/wrist/frame`, running OpenVINO CPU inference (`OpenVINODetector`), publishing to
    `so101/perception/wrist_anomaly`, optionally appending JSONL events, and posting
    `camera="wrist"` updates via `CamUiReporter`.
  - `services/wrist_anomaly_detector/wrist_anomaly_node.py`: standalone BabyROS daemon defaulting
    to `/home/intel/brickcell-deploy/artifacts/models/plate-calibrated-707-v1/model.xml`,
    `http://127.0.0.1:8091/frame.jpg`, and `http://127.0.0.1:8093`.
  - `src/brickcell/block_inspector/camera_adapters.py`, `src/brickcell/block_inspector/hardware_cell.py`,
    `scripts/publish_camui_detection.py`, `scripts/publish_mark_inspection.py`,
    `scripts/deploy_to_intel.ps1`, `tests/test_camui_detection.py`.
- **Live runtime verification on `intel` (`127.0.0.1:8093`)**:
  - `camui.service` (`systemd --user`, PID `4001583`) restarted cleanly serving
    `overhead` (`Top`), `side` (`Plate`), and `wrist` (`Gripper`).
  - `tmux -L camui-publisher` (`pub`), `tmux -L camui-detection-bridge` (`bridge`), and
    `tmux -L babyros-wrist` (`wrist`) running continuously with no errors in `pub.log` or
    `bridge.log`.
  - `GET /api/status` and `GET /api/detection` polled at `t=0` and `t=+2s` confirmed both
    `cameras.side` (`GOOD | PatchCore 2.265 / 2.225 vs thr 2.454 (6d8add3e0688a925)`,
    `age_s = 0.85s / 0.80s`) and `cameras.wrist` (`DEFECT FLAGGED | Wrist PatchCore 2.582 / 2.584 vs thr 2.454 (6d8add3e0688a925)`,
    `age_s = 0.12s / 0.21s`) updating concurrently without clobbering each other.
  - `wrist-events.jsonl` recorded structured events with
    `score_type = "raw_patchcore_distance_not_probability"`, `crop = [0, 0, 640, 360]`, and
    `model_version = "6d8add3e0688a925"`.
- **Executed verification checks on `intel` (`/home/intel/brickcell-deploy/.venv`) & Windows host**:
  - `.venv/bin/ruff check src tests scripts services/wrist_anomaly_detector camera-ui/camui.py`: `All checks passed!`.
  - `.venv/bin/ruff format --check src tests scripts services/wrist_anomaly_detector camera-ui/camui.py`: `139 files already formatted`.
  - `.venv/bin/pytest -v tests/test_camui_detection.py tests/test_heuristic_sort_mode.py tests/test_mark_scan.py tests/test_mark_bundle.py tests/test_inspection.py`: `75 passed in 2.51s`.
  - `MUJOCO_GL=egl .venv/bin/pytest -v tests/test_block_inspector.py -k test_wrist_anomaly_babyros_node_publishes_anomaly_alerts`: `1 passed, 19 deselected in 1.28s`.
  - `node camera-ui/test_ui.cjs` (Windows host): `Camera red warning and stale/offline reset checks passed`.
- **Skipped checks & limitations**:
  - `camera-ui/test_ui.cjs` could not run directly on `intel` because `node` is not installed there.
  - Full `pytest` suite stops on `tests/test_foreground_memory.py` in `/home/intel/brickcell-deploy/.venv`
    (`No module named 'torch'`) because that runtime venv excludes training dependencies.
  - `plate-calibrated-707-v1` was trained and calibrated on `plate` (`243622060450`) scenes, not a
    `wrist` (`sonix-wrist`) dataset; as a result, on the live `wrist` background it scores
    ~`2.58` (above the `plate` threshold `2.454`) until calibrated on dedicated `wrist` good/defective
    samples.
  - Note in `wrist.log`: Python 3.12 emitted a `RuntimeWarning` about `os.fork()` with multithreaded
    JAX when BabyROS initializes; events and `camui` updates continued normally.
