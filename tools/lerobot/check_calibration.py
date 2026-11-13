"""Read-only: show the leader calibration and compare leader and follower poses (calibrated values)."""
import json
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00"
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00"
r = make_robot_from_config(SO101FollowerConfig(port=F, id="follower"))
t = make_teleoperator_from_config(SO101LeaderConfig(port=L, id="leader"))
for n, c in t.calibration.items():
    print(f"leader {n:14} offset={c.homing_offset:6} min={c.range_min:5} max={c.range_max:5}")
out, raw = {}, {}
for name, dev in (("follower", r), ("leader", t)):
    dev.bus.connect()
    try:
        out[name] = dev.bus.sync_read("Present_Position")
        raw[name] = dev.bus.sync_read("Present_Position", normalize=False)
    finally:
        dev.bus.disconnect(disable_torque=False)
for j in out["leader"]:
    lv, fv = out["leader"][j], out["follower"][j]
    print(f"{j:14} leader={lv:7.1f} (raw {raw['leader'][j]:4}) follower={fv:7.1f} (raw {raw['follower'][j]:4}) diff={lv - fv:6.1f}")
