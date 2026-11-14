# Future Intel cell deployment — not yet hardware-validated

Target supplied by the challenge brief: Ubuntu 26.04 on Intel Core Ultra X9 Series 3 (388H), SO-101 follower/leader arms, wrist camera, D415 and QHD webcam. This prototype has only run on Apple M4 Max. No Intel GPU/NPU, physical camera/arm, Studio UI, VLA training job, or Modal workload has been exercised.

## Transfer and establish CPU correctness

Copy the repository and trusted model bundle (`model.xml`, `model.bin`, `model.json`) through the team's approved transfer mechanism. Do not transfer private keys or arbitrary development caches. Keep original checkpoint, manifests and reports for provenance. Model XML/BIN hashes are verified by the loader.

Install native Python 3.12/uv on the Intel laptop, then from the repository:

```bash
uv sync --locked
uv run python -m brickcell doctor
uv run pytest
uv run python -m brickcell predict --model artifacts/models/patchcore/model.xml --image PATH_TO_LOCAL_IMAGE
```

The existing lock includes Linux x86_64 resolution and the official CPU PyTorch index. Installation and device drivers on Ubuntu 26.04 remain to be tested. Do not assume that a device name appearing in the challenge brief is available to OpenVINO.

```bash
uv run python -c 'import openvino as ov; c=ov.Core(); print(ov.__version__); print(c.available_devices)'
uv run python -c 'import openvino as ov; m=ov.Core().read_model("artifacts/models/patchcore/model.xml"); print([(p.any_name,str(p.partial_shape)) for p in m.inputs]); print([(p.any_name,str(p.partial_shape)) for p in m.outputs])'
```

The verified baseline shape is static `[1,3,128,128]`; output shapes are score `[1]` and map `[1,1,128,128]`. If the model is retrained at another resolution, verify the new metadata. Run the same known validation images and numerical parity on CPU before device changes. Recalibrate against real camera data: the synthetic threshold is not suitable for deployment.

## GPU/NPU compatibility and optimization

Consult the installed [OpenVINO version's device documentation](https://docs.openvino.ai/2026/) for OS/driver support. `Core.available_devices` is authoritative for that laptop. Probe `query_model` and compile on each available explicit device; compilation success alone does not establish output parity or throughput. PatchCore nearest-neighbor/memory-bank operations can have device-specific support and memory constraints. If a device cannot execute the graph, keep the working CPU path and record the exact unsupported operation/error. Do not label an AUTO/HETERO fallback as pure NPU execution.

Brickcell intentionally fixes baseline inference to CPU. Create a separately reviewed runtime option for alternate devices after validation; do not silently change it. With OpenVINO's installed benchmark application, inspect `uv run benchmark_app -h` before invoking its version-specific flags. Typical separate measurements are:

```bash
uv run benchmark_app -m artifacts/models/patchcore/model.xml -d CPU -hint latency -t 10
# Only after device discovery and successful compilation:
uv run benchmark_app -m artifacts/models/patchcore/model.xml -d GPU -hint latency -t 10
uv run benchmark_app -m artifacts/models/patchcore/model.xml -d NPU -hint latency -t 10
```

These are future commands, not executed Intel benchmarks. Compare synchronous latency and asynchronous throughput separately. Record warmup, sample count, compile time, power mode, streams, shape, memory-bank size, precision and full device name. Measure decode/preprocess/inference/control latency separately and together. The current `brickcell benchmark` provides CPU inference-only p50/p95 for a fixed image.

First retain the uncompressed f32 export as the reference. Next export FP16 to a new directory and repeat validation parity/threshold checks on each intended device. Consider INT8 only later using NNCF with representative **calibration** images, never the final test set. Quantization may change score distributions; version/recalibrate the threshold against validation data and use a fresh held-out test after selecting the variant. Preserve the full-precision rollback bundle.

## Cameras

Use a separately versioned Physical AI Runtime environment; its current dependency requirements conflict with parts of the Anomalib environment. Exchange frames/results through explicit schemas or a local process boundary. Its [camera APIs](https://github.com/openvinotoolkit/physicalai#camera-api) expose discovery and UVC/RealSense implementations; verify the installed signatures at deployment.

Discover actual device IDs and serial numbers. Check `/dev/video*` and USB permissions on Ubuntu; do not assume indices persist across reconnects. Confirm D415 firmware, librealsense/pyrealsense2 support, RGB/depth alignment, frame sizes, timestamps and depth units. Use serial-based selection for RealSense and stable device paths where possible. The wrist and QHD webcams need separate color/size/exposure checks.

Implement `FrameSource.capture(object_id, attempt)` with acquisition time after the presentation/settling action and a unique frame token. Drop buffered pre-movement frames. Convert OpenCV BGR to RGB exactly once, test with known red/blue targets, and preserve the model's resize/normalization contract. Pin exposure/focus/white balance where practical. Depth may help presence/pose checks, but it does not substitute for calibrated RGB defect inspection. Add camera disconnect, stale timestamp, timeout and dropped-frame tests before unattended runs.

## SO-101 and policy arbitration

The [Physical AI Runtime robot protocol](https://github.com/openvinotoolkit/physicalai#robot-api) exposes low-level observations and actions. Brickcell's interface is intentionally task-level (`start_pick`, inspection pose, bowl motion, release, completion). A new `SO101RobotAdapter` must plan/execute these actions through the installed driver or a learned policy and return unique action tokens. Reaching a commanded joint position does not prove a grasp or release.

Before motion, identify follower/leader serial ports and calibrations; confirm joint ordering, signs, units, gripper range, torque/velocity limits, workspace boundaries and camera extrinsics. The tested simulation reports normalized body joints and gripper values; do not assume hardware drivers/policies use the same convention. Check model metadata and driver observations explicitly. Never replay normalized values as radians/degrees.

Provide a physical emergency stop, hardware watchdog, safe recovery poses, joint/velocity limits and an operator-controlled enable before any powered test. `stop_safely` must map to a real stop, independent of logging/inference failures. Add sensor-backed grasp/presence/release verification; model an ambiguous release as an error requiring recovery. Start with restricted speed and an empty workspace under supervision.

Give one component exclusive ownership of motor commands. An ACT/SmolVLA policy may control the pick/present subtask. On transition to inspection, cancel/drain queued policy actions, hold a verified safe pose, reset policy state as appropriate, and capture a fresh frame. Only a valid current inspection result can choose a bowl action. Prevent the policy loop and supervisor from commanding the arm concurrently. Preserve stale-result rejection, bounded retries and timeouts at every boundary.

## Studio, demonstrations and Modal later

Use [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) once the station and leader/follower setup are available. Confirm supported ACT/SmolVLA policies, observation keys, action dimensions, units, frame rate and export format in the station's installed version. Collect diverse successful pick/present/release demonstrations and representative recoveries. Keep demonstration train/validation/test episodes grouped by session/objects. Inspection dataset labels and manipulation demonstrations are distinct assets.

When organizers supply Modal workspace access, GPU allocation and training instructions, create a separate pinned training environment from the verified Studio recipe. Authenticate through the approved Modal flow; keep credentials in the platform's secrets mechanism, never this repository. Mount approved demonstration datasets and checkpoints in a persistent volume, record versions/config/seed, and start with a small job within the supplied budget. Preserve checkpoints/logs and evaluate on held-out demonstrations before exporting the policy. No cloud command is included here because the actual workspace, policy recipe, budget and credentials are not yet known. This task submitted no jobs and used no unrelated credentials.

Load the exported policy in Physical AI Runtime after checking its observation/action contract, first in simulation then supervised hardware. A successful software mock cycle or MuJoCo scene is not a physical safety or task-success validation.
