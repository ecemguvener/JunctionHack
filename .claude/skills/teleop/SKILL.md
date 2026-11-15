---
name: teleop
description: Start, stop, or check leader -> follower teleoperation of the two SO-101 arms on the Intel server with one command. Use when the user asks to set up teleop / teleoperation, make the follower follow the leader, mirror the arms, or stop teleop.
argument-hint: "[start|status|attach|stop]"
---

# Teleoperation of the SO-101 arms

The user wants the follower arm to follow the leader arm. Use the one-shot script `tools/lerobot/teleop.sh`. Run it on this computer. It connects to the Intel server through SSH (alias `intel`, see [ssh.md](../../../ssh.md)).

## Actions

| Command | What it does |
|---|---|
| `tools/lerobot/teleop.sh start` | Releases both arms in armd, makes sure that the serial ports are free, and starts `tools/lerobot/teleop_aligned.py` in tmux (`tmux -L teleop`, session `teleop`). It shows the first alignment screen. |
| `tools/lerobot/teleop.sh status` | Shows if teleoperation runs, which process has each port, the armd health, and the last screen. |
| `tools/lerobot/teleop.sh stop` | Sends Ctrl+C to teleoperation, stops the tmux server, and acquires both arms in armd again. |
| `tools/lerobot/teleop.sh attach` | Shows the live screen. It needs a terminal, so give this command to the user. Do not run it yourself. |

Use the argument from the user. If there is no argument, use `start`.

Settings: `ALIGN_LIMIT` (default 20 degrees), `MAX_STEP` (default 30 degrees), `TELEOP_HOST` (default `intel`), `TELEOP_DIR` (default `~/Desktop/tarot-club-junction`). Do not set `MAX_STEP` to 2 or 5. With a small step limit, the follower is far behind the leader.

## Procedure

1. Run the script with the action. If SSH fails, follow [ssh.md](../../../ssh.md) step 2.
2. After `start`, tell the user these items:
   - The follower does not move until the user moves the **leader** by hand to the follower pose. Each joint must show `OK` (within `ALIGN_LIMIT`) for 1 s. Give the table of joints that are not `OK`, with the hint from the screen.
   - The command to see the live screen: `tools/lerobot/teleop.sh attach`.
   - To stop: `tools/lerobot/teleop.sh stop`. **Hold the follower** before the stop, because LeRobot turns off the follower torque.
3. If the script shows `ERROR`, report the error and the log lines. Do not try other ways to open the ports. Do not stop armd or other processes that the script did not start, unless the user asks.

## Limits

- You cannot move the leader. Do not tell the user that the follower follows until the screen shows that the alignment is complete and `lerobot-teleoperate` runs.
- If something calls `acquire` in armd while teleoperation runs, teleoperation can stop. Use `status` to find this, and tell the user.
