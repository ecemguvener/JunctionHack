"""WRITES SERVO EEPROM. Reverse servo IDs (1<->6, 2<->5, 3<->4) via temporary IDs. Stops on first failure.

This was a ONE-TIME fix (2026-11-13): the leader arm had its IDs in reverse order (ID 1 = gripper).
It is done. If you run it again, it reverses the IDs back to the WRONG order.
Run it only if scan_servos.py and identify_servos.py show that ID 1 is the gripper again.
After it runs, calibrate the leader again.

Usage: python renumber_leader_ids.py <port> --yes-reverse-ids
"""
import sys, time
import scservo_sdk as s
if "--yes-reverse-ids" not in sys.argv:
    sys.exit("Refused: read the docstring first. This writes servo EEPROM. Add --yes-reverse-ids to run.")
port = sys.argv[1]
ph = s.PortHandler(port); assert ph.openPort() and ph.setBaudRate(1_000_000)
pk = s.sms_sts(ph)
alive = lambda i: pk.ping(i)[1] == s.COMM_SUCCESS
before = [i for i in range(21) if alive(i)]
assert before == [1, 2, 3, 4, 5, 6], f"unexpected IDs before: {before}"
pos_before = {i: pk.read2ByteTxRx(i, 56)[0] for i in before}
steps = [(1, 11), (6, 1), (11, 6), (2, 12), (5, 2), (12, 5), (3, 13), (4, 3), (13, 4)]
for old, new in steps:
    assert alive(old) and not alive(new), f"precheck failed {old}->{new}"
    res, err = pk.write1ByteTxRx(old, 5, new)
    time.sleep(0.05)
    ok = alive(new) and not alive(old)
    print(f"ID {old:2} -> {new:2}: write res={res} err={err} verify={'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("STOPPED. Present IDs:", [i for i in range(21) if alive(i)]); sys.exit(1)
after = [i for i in range(21) if alive(i)]
print("IDs after:", after)
mapping = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
for o, n in mapping.items():
    print(f"  old ID{o} pos={pos_before[o]:4}  ->  new ID{n} pos={pk.read2ByteTxRx(n, 56)[0]:4}")
ph.closePort()
