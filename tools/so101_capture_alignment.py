"""Capture a human-matched CAD pose using READ packets, then save its alignment.

Run only after the person has matched the displayed target and is holding still.
No motor commands, calibration-register writes, or service changes are made.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from so101_calibration import ServoCalibration, MOTOR_NAMES, ALIGNMENT_PATH
from so101_kinematics import SO101Kinematics, CalibratedSO101Kinematics, ARM_JOINTS
from so101_readonly_snapshot import snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('leader', 'follower'), required=True)
    parser.add_argument('--target', type=Path, default=ROOT/'calibration/reference_poses/upright-cad-target.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirm-visual-match', action='store_true')
    args = parser.parse_args()
    if not args.confirm_visual_match:
        parser.error('Capture requires a human-confirmed visual match')
    if args.output.exists():
        parser.error('The output already exists; use a new measurement filename')
    target = json.loads(args.target.read_text(encoding='utf-8'))
    if target['model_sha256'] != hashlib.sha256((ROOT/'SO-101.zip').read_bytes()).hexdigest():
        raise ValueError('CAD target does not match the current model archive')
    if target['joint_order'] != list(ARM_JOINTS):
        raise ValueError('Unexpected CAD target joint order')
    q_cad = np.asarray(target['q_cad_rad'], dtype=float)
    if q_cad.shape != (5,) or not np.isfinite(q_cad).all():
        raise ValueError('Target must contain five finite CAD angles')
    original = json.loads(ALIGNMENT_PATH.read_text(encoding='utf-8'))
    profile = original['arms'][args.arm]
    cal = ServoCalibration(args.arm)
    if profile['calibration_sha256'] != cal.fingerprint:
        raise ValueError('Alignment does not match the servo calibration')
    signs = np.array([profile['cad_sign'][n] for n in MOTOR_NAMES], dtype=float)
    if not np.isin(signs, [-1, 1]).all():
        raise ValueError('All five joint direction signs must be recorded first')
    measured = snapshot((args.arm,))
    ticks = np.array([measured['arms'][args.arm]['joints'][n]['raw'] for n in MOTOR_NAMES])
    q_encoder = cal.validate_limits(cal.ticks_to_radians(ticks))
    offset = q_cad-signs*q_encoder
    offset = (offset+np.pi) % (2*np.pi)-np.pi
    candidate = copy.deepcopy(original)
    updated = candidate['arms'][args.arm]
    updated['cad_at_zero_rad'] = dict(zip(MOTOR_NAMES, offset.tolist()))
    updated['confirmed'] = True
    updated['reference_method'] = 'human-confirmed visual match to a specified CAD pose'
    updated['reference_uncertainty'] = 'not quantified; this is not a metrology calibration'
    updated['reference_captured_utc'] = measured['timestamp_utc']
    updated['reference_model_sha256'] = target['model_sha256']
    updated['reference_evidence'] = str(args.output.resolve().relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix='so101-capture-check-') as tmp:
        check = Path(tmp)/'alignment.json'
        check.write_text(json.dumps(candidate), encoding='utf-8')
        model = CalibratedSO101Kinematics(args.arm, alignment_path=check)
        np.testing.assert_allclose(model.fk(q_encoder), SO101Kinematics().fk(q_cad), atol=1e-12)
        model.export_arm_urdf(Path(tmp)/'checked.urdf')
    evidence = {
        'version': 1, 'arm': args.arm,
        'method': updated['reference_method'],
        'uncertainty': updated['reference_uncertainty'],
        'cad_target': target, 'snapshot': measured,
        'q_encoder_rad': q_encoder.tolist(),
        'cad_at_zero_rad': offset.tolist(), 'cad_sign': signs.tolist(),
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'servo_writes': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        f.write(json.dumps(evidence, indent=2)+'\n')
    ALIGNMENT_PATH.write_text(json.dumps(candidate, indent=2)+'\n', encoding='utf-8')
    models = ROOT/'models'
    models.mkdir(exist_ok=True)
    model.export_arm_urdf(models/f'so101-{args.arm}-calibrated.urdf')
    print('ARM', args.arm)
    print('RAW', dict(zip(MOTOR_NAMES, ticks.tolist())))
    print('OFFSETS_RAD', updated['cad_at_zero_rad'])
    print('CONFIRMED_MANUAL_VISUAL_ALIGNMENT', updated['confirmed'])
    print('No servo registers were written.')


if __name__ == '__main__':
    main()
