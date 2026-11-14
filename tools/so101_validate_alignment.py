"""Offline checks of captured arm alignments against Frax and finite differences.

Run with the installed Frax interpreter and JAX_ENABLE_X64=1. No hardware access.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from so101_kinematics import CalibratedSO101Kinematics, SO101Kinematics, ARM_JOINTS
from so101_calibration import ALIGNMENT_PATH

ROOT = Path(__file__).resolve().parents[1]


def validate(arm):
    alignment = json.loads(ALIGNMENT_PATH.read_text(encoding='utf-8'))['arms'][arm]
    evidence = json.loads((ROOT/alignment['reference_evidence']).read_text(encoding='utf-8'))
    model = CalibratedSO101Kinematics(arm)
    robot = model.load_frax()
    assert robot.nv == 5
    reference = np.array(evidence['q_encoder_rad'])
    target = np.array(evidence['cad_target']['q_cad_rad'])
    np.testing.assert_allclose(model.fk(reference), SO101Kinematics().fk(target), atol=1e-12)
    rng = np.random.default_rng(101)
    poses = [reference, np.zeros(5), *rng.uniform(model.calibration.lower*.6,
                                                model.calibration.upper*.6, (8, 5))]
    max_pose = max_jacobian = max_finite_difference = 0.
    for q in poses:
        max_pose = max(max_pose, float(np.max(np.abs(np.asarray(robot.ee_transform(q))-model.fk(q)))))
        max_jacobian = max(max_jacobian, float(np.max(np.abs(np.asarray(robot.ee_jacobian(q))-model.jacobian(q)))))
        h = 1e-6
        columns = []
        rotation = model.fk(q)[:3, :3]
        for i in range(5):
            dq = np.zeros(5)
            dq[i] = h
            plus, minus = model.fk(q+dq), model.fk(q-dq)
            skew = ((plus[:3, :3]-minus[:3, :3])/(2*h))@rotation.T
            columns.append(np.r_[(plus[:3, 3]-minus[:3, 3])/(2*h),
                                 [skew[2, 1], skew[0, 2], skew[1, 0]]])
        max_finite_difference = max(max_finite_difference,
                                    float(np.max(np.abs(model.jacobian(q)-np.array(columns).T))))
    assert max_pose < 2e-8 and max_jacobian < 2e-8 and max_finite_difference < 2e-8
    path = ROOT/f'models/so101-{arm}-calibrated.urdf'
    joints = {j.get('name'): j for j in ET.parse(path).getroot().findall('joint')}
    for i, name in enumerate(ARM_JOINTS):
        assert joints[name].get('type') == 'revolute'
        limit = joints[name].find('limit')
        np.testing.assert_allclose([float(limit.get('lower')), float(limit.get('upper'))],
                                   [model.calibration.lower[i], model.calibration.upper[i]], atol=1e-14)
    report = {
        'arm': arm, 'reference_method': alignment['reference_method'],
        'measurement_uncertainty': alignment['reference_uncertainty'],
        'poses_checked': len(poses), 'max_frax_pose_entry_error': max_pose,
        'max_frax_jacobian_entry_error': max_jacobian,
        'max_finite_difference_jacobian_error': max_finite_difference,
        'position_limits_match_calibration': True, 'physical_accuracy_measured': False,
        'hardware_access_performed': False,
    }
    (ROOT/f'models/so101-{arm}-validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('both', 'leader', 'follower'), default='both')
    args = parser.parse_args()
    for arm in ('leader', 'follower') if args.arm == 'both' else (args.arm,):
        print(json.dumps(validate(arm), indent=2))


if __name__ == '__main__':
    main()
