"""Read-only: show voltage, temperature, torque, status and position of servo IDs 1-6 on one bus.
Usage: python servo_status.py <port>"""
import sys, time
import scservo_sdk as s
ph = s.PortHandler(sys.argv[1]); assert ph.openPort() and ph.setBaudRate(1_000_000)
pk = s.sms_sts(ph)
for i in range(1, 7):
    m, res, err = pk.ping(i)
    v = pk.read1ByteTxRx(i, 62)[0]; t = pk.read1ByteTxRx(i, 63)[0]; st = pk.read1ByteTxRx(i, 65)[0]
    tq = pk.read1ByteTxRx(i, 40)[0]; pos = pk.read2ByteTxRx(i, 56)[0]; cur = pk.read2ByteTxRx(i, 69)[0]
    prot = pk.read1ByteTxRx(i, 19)[0]; maxtq = pk.read2ByteTxRx(i, 16)[0]
    print(f"ID{i} ping_res={res} err_bits={err} status={st:08b} volt={v/10} temp={t} torque={tq} pos={pos} current={cur} unload_cond={prot:08b} max_torque={maxtq}")
ph.closePort()
