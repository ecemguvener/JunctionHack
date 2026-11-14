"""Read-only: ping IDs 0-20 on a bus; show model, lock, torque, position."""
import sys
import scservo_sdk as s
port = sys.argv[1]
ph = s.PortHandler(port); assert ph.openPort() and ph.setBaudRate(1_000_000)
pk = s.sms_sts(ph) if hasattr(s, "sms_sts") else s.PacketHandler(0)
found = []
for i in range(0, 21):
    model, res, err = pk.ping(i)
    if res == s.COMM_SUCCESS:
        lock, _, _ = pk.read1ByteTxRx(i, 55)
        tq, _, _ = pk.read1ByteTxRx(i, 40)
        pos, _, _ = pk.read2ByteTxRx(i, 56)
        found.append(i); print(f"ID{i:2} model={model} lock={lock} torque={tq} pos={pos}")
print("FOUND", found)
ph.closePort()
