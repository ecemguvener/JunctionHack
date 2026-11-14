# LeRobot helper scripts for the two SO-101 arms

These scripts were used on 2026-11-13 to calibrate the two arms and to start teleoperation with LeRobot 0.7.0. The calibration files are in [../../calibration](../../calibration).

Run them on the Intel server, in the uv project, with the project Python:

```bash
cd ~/Desktop/tarot-club-junction
.venv/bin/python /path/to/tools/lerobot/<script>.py
```

The serial ports are set in the scripts (follower `...5B3D040991...`, leader `...5AE7044736...`). Only one process can open a port. Stop armd and LeRobot before you run a script (see [arm-api.md](../../arm-api.md), section 12).

## Scripts

| Script | Writes to servos? | What it does |
|---|---|---|
| `teleop_aligned.py` | Only after alignment (LeRobot teleoperation) | Shows the leader and follower poses live. When all joints are within 20° for 1 s, it starts `lerobot-teleoperate` with a 30° step limit. Settings: `ALIGN_LIMIT`, `MAX_STEP` environment variables. |
| `compare_arms.py` | No | Shows the leader and follower values side by side. Move the same joint on both arms in the same direction: it shows `SAME direction` or `OPPOSITE` (inverted joint). |
| `identify_servos.py` | No | Shows which servo ID moves when you move a joint. Optional argument: port (default: leader). |
| `check_calibration.py` | No | Shows the leader calibration and the difference between the leader and follower poses. |
| `scan_servos.py` | No | Pings IDs 0–20 on a port. Shows model, lock, torque and position. Argument: port. |
| `servo_status.py` | No | Shows voltage, temperature, torque, status and position of IDs 1–6. Argument: port. |
| `renumber_leader_ids.py` | **Yes (EEPROM)** | One-time fix: reverses the servo IDs (1↔6, 2↔5, 3↔4). **Do not run it again**: the leader IDs are correct now. It needs `--yes-reverse-ids`. |

## Start teleoperation

1. Make sure that no other process uses the ports.
2. Run `teleop_aligned.py`. Move the leader until all joints show `OK`. Teleoperation then starts.
3. To stop, push Ctrl+C. LeRobot turns off the follower torque. Hold the follower arm.

Do not use a small step limit (`MAX_STEP` 2 or 5). With a small limit, the follower is far behind the leader, and the movement can look inverted.
