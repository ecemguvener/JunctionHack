"""Read-only: live side-by-side of leader vs follower (calibrated values). Ctrl+C to stop."""
import time
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00"
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00"
fb = make_robot_from_config(SO101FollowerConfig(port=F, id="follower")).bus
lb = make_teleoperator_from_config(SO101LeaderConfig(port=L, id="leader")).bus
fb.connect(); lb.connect()
names = list(fb.motors)
f0 = fb.sync_read("Present_Position"); l0 = lb.sync_read("Present_Position")
print("Move the SAME joint on BOTH arms in the SAME physical direction.")
print("If the two 'moved' numbers have OPPOSITE signs, that joint is inverted. Ctrl+C to stop.\n")
for n in names: print()
mov = {n: [0.0, 0.0] for n in names}
try:
    while True:
        f = fb.sync_read("Present_Position"); l = lb.sync_read("Present_Position")
        out = [f"\033[{len(names)}A"]
        for n in names:
            dl, df = l[n] - l0[n], f[n] - f0[n]
            if abs(dl) > abs(mov[n][0]): mov[n][0] = dl
            if abs(df) > abs(mov[n][1]): mov[n][1] = df
            flag = ""
            if abs(dl) > 10 and abs(df) > 10:
                flag = "SAME direction" if (dl > 0) == (df > 0) else "OPPOSITE  <-- inverted"
            out.append(f"{n:14} leader={l[n]:7.1f} moved={dl:+7.1f} | follower={f[n]:7.1f} moved={df:+7.1f}  {flag:24}")
        print("\n".join(out), flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    fb.disconnect(disable_torque=False); lb.disconnect(disable_torque=False)
print("\nSUMMARY (largest move seen, leader vs follower):")
for n in names:
    dl, df = mov[n]
    v = "-" if abs(dl) < 10 or abs(df) < 10 else ("SAME" if (dl > 0) == (df > 0) else "OPPOSITE")
    print(f"  {n:14} leader={dl:+7.1f} follower={df:+7.1f}  {v}")
