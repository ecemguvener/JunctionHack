"""Read-only: show which leader servo ID moves. Ctrl+C to stop."""
import sys, time
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.teleoperators import make_teleoperator_from_config
port = sys.argv[1] if len(sys.argv) > 1 else "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00"
b = make_teleoperator_from_config(SO101LeaderConfig(port=port, id="leader")).bus
b.connect()
names = list(b.motors)
ids = {n: b.motors[n].id for n in names}
read = lambda: b.sync_read("Present_Position", normalize=False)
start = read(); prev = dict(start); unw = {n: 0 for n in names}; span = {n: [0, 0] for n in names}
print("Move ONE joint at a time. The ID that moves shows the largest MOVED value. Ctrl+C to stop.\n", flush=True)
try:
    last = 0
    while True:
        cur = read()
        for n in names:
            d = cur[n] - prev[n]
            if d > 2048: d -= 4096
            elif d < -2048: d += 4096
            unw[n] += d; prev[n] = cur[n]
            span[n][0] = min(span[n][0], unw[n]); span[n][1] = max(span[n][1], unw[n])
        if time.time() - last > 0.25:
            line = "  ".join(f"ID{ids[n]}:{unw[n]*360/4096:+6.0f}" for n in names)
            print(f"now vs start (deg)  {line}   ", end="\r", flush=True); last = time.time()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    b.disconnect(disable_torque=False)
print("\nSUMMARY total range seen per ID (deg):")
for n in names:
    print(f"  ID{ids[n]} (LeRobot name {n:14}) range={(span[n][1]-span[n][0])*360/4096:6.1f}")
