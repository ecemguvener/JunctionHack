"""Read-only SO-101 position/torque snapshot for manual sign measurements.

Uses only SDK READ instructions. No LeRobot connection/calibration callbacks,
servo register writes, torque changes, or service lifecycle operations are used.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS

from so101_calibration import ServoCalibration, MOTOR_NAMES

PORTS = {
    'follower': '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00',
    'leader': '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00',
}
TORQUE_ENABLE = 40
PRESENT_POSITION = 56


def snapshot(arms=None):
    out = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'read_only': True,
        'units': {'raw': 'homed Present_Position ticks',
                  'arm_value': 'LeRobot calibrated degrees',
                  'gripper_value': 'calibration range percent; physical open direction unverified'},
        'arms': {},
    }
    selected = tuple(PORTS) if arms is None else tuple(arms)
    if not selected or any(arm not in PORTS for arm in selected):
        raise ValueError('Select follower and/or leader')
    for arm in selected:
        port_name = PORTS[arm]
        owner = subprocess.run(['fuser', port_name], capture_output=True)
        if owner.returncode != 1:
            raise RuntimeError(f'{arm}: port is occupied or ownership check failed')
        cal = ServoCalibration(arm)
        port = PortHandler(port_name)
        try:
            if not port.openPort() or not port.setBaudRate(1000000):
                raise RuntimeError(f'{arm}: cannot open serial port')
            sdk = sms_sts(port)
            joints = {}
            for name in (*MOTOR_NAMES, 'gripper'):
                sid = cal.records[name]['id']
                torque, status, error = sdk.read1ByteTxRx(sid, TORQUE_ENABLE)
                if status != COMM_SUCCESS or error:
                    raise RuntimeError(f'{arm}/{name}: torque read failed: {status}, {error}')
                if torque != 0:
                    raise RuntimeError(f'{arm}/{name}: torque is enabled; no state was changed')
                raw, status, error = sdk.read2ByteTxRx(sid, PRESENT_POSITION)
                if status != COMM_SUCCESS or error or not 0 <= raw <= 4095:
                    raise RuntimeError(f'{arm}/{name}: invalid position read: {raw}, {status}, {error}')
                entry = cal.records[name]
                if name == 'gripper':
                    value = cal.gripper_ticks_to_percent(raw)
                else:
                    value = (raw-(entry['range_min']+entry['range_max'])/2)*360/4095
                joints[name] = dict(id=sid, raw=raw, value=value,
                                    raw_degrees=raw*360/4096, torque_enabled=False)
            out['arms'][arm] = {'port': port_name, 'calibration_sha256': cal.fingerprint,
                                'joints': joints}
        finally:
            port.closePort()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', choices=('both', 'follower', 'leader'), default='both')
    args = parser.parse_args()
    result = snapshot(None if args.arm == 'both' else (args.arm,))
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            f.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
