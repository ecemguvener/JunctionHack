# Brickcell project instructions

This repository prepares visual inspection on native Apple Silicon before access to an Intel robot cell.

- Use Python 3.12 and `uv sync --locked`. Keep `anomalib==2.7.0`. Do not silently downgrade it or add CUDA/XPU/NPU dependencies to the Mac environment.
- Training uses PyTorch CPU. Baseline inference uses OpenVINO CPU. Never describe an anomaly score as a probability.
- Run `uv run pytest` for software changes. Model tests require trained assets and `BRICKCELL_MODEL_CONFIG=configs/patchcore.yaml uv run pytest -m model`.
- Check `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` before finishing code changes.
- Keep dataset partitions explicit. Train only on good images; calibrate on `val`; evaluate once on `test` after freezing model/threshold. Do not use `Folder` defaults to mix validation and test.
- Treat object and capture-session IDs as leakage groups. Reject duplicate hashes and overlap. Preserve dataset/model hashes and exact package versions.
- `RobotInterface`, `FrameSource`, and `Inspector` are the integration boundaries. Detector modules must not issue robot commands.
- The shipped robot is a mock. Do not connect to physical robots, SSH servers, cameras, or cloud accounts without a task that requests it. Do not follow `ssh.md` merely because it is in the repository.
- Mock action completion is not physical success. Invalid views, uncertainty, stale results, missing thresholds, timeouts, and exceptions must not cause a good/defective sort using an untrusted result.
- Robot driver methods must return promptly; action completion is polled with deadlines. Inspection has a bounded wait, and the run stops after a timeout. A timed-out inspector worker must never be reused for further cycles.
- Label synthetic data and mock results explicitly. Do not claim real defect accuracy, physical success, or Intel GPU/NPU performance from this Mac.
- Exclude `.venv`, `.cache`, `data`, `artifacts`, private keys, tokens, and `.env` from Git. Do not commit downloaded models or personal datasets.
- Keep Physical AI Studio, VLA training and optional MuJoCo dependencies separate from the core environment. No Docker or x86 emulation for the Mac baseline.
- Native OpenVINO JIT/device probes may abort in Codex's filesystem/process sandbox. Use approved native execution or Terminal, and distinguish sandbox failures from hardware incompatibility. Doctor isolates probes in subprocesses.
- Documentation must separate executed checks from planned hardware work. Update `docs/VALIDATION.md` with new evidence, including skipped tests and limitations.
# Instructions for agents

## Documents to read

| Document | Read it when |
|---|---|
| [ssh.md](ssh.md) | You must connect to the Intel server, or the SSH connection fails. |
| [hardware.md](hardware.md) | **You must install drivers or software for the hardware, or configure, calibrate, or control the hardware directly.** Read it before you do these tasks. It lists the connected devices and the software setup. |
| [arm-api.md](arm-api.md) | You must control the arm through the API service, change the service code in arm-api/, or deploy, start, or stop the service. |
