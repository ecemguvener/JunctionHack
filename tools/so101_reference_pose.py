"""Validate a measured follower snapshot and generate a provisional visual fit.

Run on Intel with ~/frax/.venv/bin/python -B tools/so101_reference_pose.py.
Only wrist-roll's CAD angle is supplied by the user. The horizontal-link
constraints below are approximate visual hypotheses, not measured 3D geometry.
This script never changes confirmed alignment, URDFs, registers or robot state.
"""
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from scipy.optimize import least_squares

from so101_calibration import ServoCalibration, MOTOR_NAMES, ALIGNMENT_PATH
from so101_kinematics import SO101Kinematics

ROOT = Path(__file__).resolve().parents[1]


def main():
    alignment_before = ALIGNMENT_PATH.read_bytes()
    source = ROOT/'calibration/reference_poses/follower-arm-measurement.json'
    ref = json.loads(source.read_text(encoding='utf-8'))
    cal = ServoCalibration('follower')
    ticks = np.array([ref['motors'][n]['raw'] for n in MOTOR_NAMES])
    q_encoder = cal.ticks_to_radians(ticks)
    cal.validate_limits(q_encoder)
    values = cal.radians_to_lerobot_degrees(q_encoder)
    for i, name in enumerate(MOTOR_NAMES):
        assert ref['motors'][name]['id'] == i+1
        assert abs(values[i]-ref['motors'][name]['value']) <= .0051
    for entry in ref['motors'].values():
        assert abs(entry['raw']*360/4096-entry['raw_degrees']) <= .0051
    gripper = ref['motors']['gripper']
    assert abs(cal.gripper_ticks_to_percent(gripper['raw'])-gripper['value']) <= .0051
    assert gripper['raw'] == ref['user_pose_observations']['gripper_closed_raw']

    model = SO101Kinematics()
    home_frames = model.frames(np.zeros(5))
    structural_axes = []
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    with zipfile.ZipFile(ROOT/'SO-101.zip') as z:
        for mesh_id, link in [(0, 'Group 1'), (2, 'Body1_3')]:
            raw = z.read(f'robotics_ws_model/meshes/Arm_Design_{mesh_id}.stl')
            count = struct.unpack_from('<I', raw, 80)[0]
            vertices = np.unique(np.frombuffer(raw, dtype=dtype, count=count,
                                              offset=84)['v'].reshape(-1, 3), axis=0)
            _, _, axes = np.linalg.svd(vertices-vertices.mean(0), full_matrices=False)
            structural_axes.append((link, home_frames[link][:3, :3].T@axes[0]))

    def residual(x):
        frames = model.frames([0, *x, 0])
        # Level the long members' principal axes, not their offset pivot lines.
        # This approximates the visible members in the supplied photograph.
        # Level upper arm and forearm, horizontal wrist-roll axis. The chosen
        # solution extends toward CAD -Y; base yaw is unconstrained by the photo.
        return [*(float((frames[link][:3, :3]@axis)[2]) for link, axis in structural_axes),
                frames['Body3'][2, 2]]

    solution = least_squares(residual, np.deg2rad([170, -160, 0]),
                             xtol=1e-13, ftol=1e-13, gtol=1e-13)
    assert solution.success and np.linalg.norm(solution.fun) < 1e-8
    cad_roll = ref['user_pose_observations']['wrist_roll_cad_degrees']
    # Zero jaw articulation retains the CAD geometry's existing clearance. It
    # is not inferred from gripper percent, nor certified as contact closure.
    display = [0, *np.rad2deg(solution.x).tolist(), cad_roll, 0]
    roll_encoder = q_encoder[4]
    report = {
        'status': 'provisional_visual_fit_not_actuator_calibration',
        'reference': str(source.relative_to(ROOT)),
        'calibration_sha256': cal.fingerprint,
        'encoder_radians': dict(zip(MOTOR_NAMES, q_encoder.tolist())),
        'cad_display_degrees': dict(zip((*MOTOR_NAMES, 'gripper'), display)),
        'assumptions': [
            'Shoulder pan is zero only to choose a display orientation; actual CAD yaw is unknown.',
            'The principal axes of the upper-arm and forearm meshes are approximated as level, extended toward CAD -Y.',
            'The wrist-roll axis is approximated as horizontal.',
            'Wrist-roll CAD -90 degrees is the user-supplied approximate reference.',
            'CAD gripper angle remains zero; geometry clearance is not a measured contact model.',
        ],
        'wrist_roll_zero_offset_hypotheses_rad': {
            'if_cad_sign_positive': float(np.deg2rad(cad_roll)-roll_encoder),
            'if_cad_sign_negative': float(np.deg2rad(cad_roll)+roll_encoder),
        },
        'observed_nonoverlapping_gripper_closed_raw': gripper['raw'],
        'gripper_calibration_range_min_raw': cal.records['gripper']['range_min'],
        'alignment_confirmed': False,
    }
    out = ROOT/'visualizations/so101-reference-fit.json'
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    html = (ROOT/'visualizations/so101-kinematics.html').read_text(encoding='utf-8')
    js = (ROOT/'tools/so101_reference_view.js').read_text(encoding='utf-8')
    js = js.replace('/*REFERENCE_POSE_DEG*/', json.dumps(np.round(display, 1).tolist()))
    marker = '}catch(e){root.querySelector'
    assert html.count(marker) == 1
    html = html.replace(marker, js+'\n'+marker)
    target = ROOT/'visualizations/so101-follower-reference.html'
    target.write_text(html, encoding='utf-8')
    assert target.stat().st_size < 1_000_000
    assert ALIGNMENT_PATH.read_bytes() == alignment_before
    print('Snapshot matches the merged follower calibration within reported rounding.')
    print('CAD display angles (provisional):', np.round(display, 3))
    print('Wrist-roll zero hypotheses (degrees):',
          {k: round(float(np.rad2deg(v)), 6) for k, v in report['wrist_roll_zero_offset_hypotheses_rad'].items()})
    print('Real alignment remains unconfirmed. Rendered:', target)


if __name__ == '__main__':
    main()
