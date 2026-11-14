"""Wait until leader pose matches follower (all joints within LIMIT), then start LeRobot teleoperation.
Read-only while waiting. Ctrl+C stops at any time."""
import os, sys, time
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
LIMIT = float(os.environ.get("ALIGN_LIMIT", 20))
STEP = os.environ.get("MAX_STEP", "30")
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00"
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00"
fb = make_robot_from_config(SO101FollowerConfig(port=F, id="follower")).bus
lb = make_teleoperator_from_config(SO101LeaderConfig(port=L, id="leader")).bus
fb.connect(); lb.connect()
names = list(fb.motors)
print(f"Move the LEADER to match the FOLLOWER. Teleop starts when every joint is within {LIMIT:.0f} for 1 s.\n")
for _ in names: print()
ok_since = None
try:
    while True:
        f = fb.sync_read("Present_Position"); l = lb.sync_read("Present_Position")
        lines = [f"\033[{len(names)}A"]; worst = 0
        for n in names:
            d = l[n] - f[n]; worst = max(worst, abs(d))
            hint = "OK" if abs(d) <= LIMIT else ("move leader to LOWER value" if d > 0 else "move leader to HIGHER value")
            lines.append(f"{n:14} leader={l[n]:7.1f} follower={f[n]:7.1f} diff={d:+7.1f}  {hint:28}")
        print("\n".join(lines), flush=True)
        if worst <= LIMIT:
            ok_since = ok_since or time.time()
            if time.time() - ok_since >= 1.0: break
        else:
            ok_since = None
        time.sleep(0.1)
finally:
    fb.disconnect(disable_torque=False); lb.disconnect(disable_torque=False)
print(f"\nAligned. Starting teleoperation (max step {STEP}). Ctrl+C to stop. HOLD the follower when you stop.\n", flush=True)
exe = os.path.join(os.path.dirname(sys.executable), "lerobot-teleoperate")
os.execv(exe, [exe,
    "--robot.type=so101_follower", f"--robot.port={F}", "--robot.id=follower", f"--robot.max_relative_target={STEP}",
    "--teleop.type=so101_leader", f"--teleop.port={L}", "--teleop.id=leader"])
