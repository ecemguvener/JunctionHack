"""Calibration and coordinate transforms tested offline; no hardware access."""
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from so101_calibration import ServoCalibration, MOTOR_NAMES, ALIGNMENT_PATH
from so101_kinematics import SO101Kinematics, CalibratedSO101Kinematics, ARM_JOINTS


class CalibrationTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        for arm in ('leader', 'follower'):
            cal = ServoCalibration(arm)
            np.testing.assert_allclose(cal.ticks_to_radians(cal.tick_min), cal.lower)
            np.testing.assert_allclose(cal.ticks_to_radians(cal.tick_max), cal.upper)
            np.testing.assert_array_equal(cal.ticks_to_radians(cal.zero_ticks), np.zeros(5))
            for ticks in (cal.tick_min, cal.tick_max, np.rint(cal.zero_ticks)):
                np.testing.assert_array_equal(cal.radians_to_ticks(cal.ticks_to_radians(ticks)), ticks)
            self.assertEqual(cal.gripper_ticks_to_percent(cal.records['gripper']['range_min']), 0)
            self.assertEqual(cal.gripper_ticks_to_percent(cal.records['gripper']['range_max']), 100)
            np.testing.assert_allclose(cal.ticks_to_radians(cal.zero_ticks+1), 2*np.pi/4096)
            cal.validate_limits(cal.lower)
            cal.validate_limits(cal.upper)
            with self.assertRaises(ValueError):
                cal.validate_limits(cal.upper+1e-6)

    def test_alignment_is_not_inferred_from_homing_offset(self):
        data = json.loads(ALIGNMENT_PATH.read_text())
        with tempfile.TemporaryDirectory(prefix='so101-unconfirmed-test-') as tmp:
            path = Path(tmp)/'alignment.json'
            for arm in ('leader', 'follower'):
                data['arms'][arm]['confirmed'] = False
            path.write_text(json.dumps(data))
            for arm in ('leader', 'follower'):
                with self.assertRaisesRegex(ValueError, 'not confirmed'):
                    CalibratedSO101Kinematics(arm, alignment_path=path)

    def test_matches_installed_lerobot_degree_normalization(self):
        try:
            importlib.metadata.version('lerobot')
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('LeRobot is not installed in this interpreter')
        from lerobot.motors import MotorCalibration
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.robots import make_robot_from_config

        # Construction only: no connect(), register writes or port opening.
        bus = make_robot_from_config(SO101FollowerConfig(port='offline-test', id='follower')).bus
        for arm in ('leader', 'follower'):
            cal = ServoCalibration(arm)
            bus.calibration = {n: MotorCalibration(**entry) for n, entry in cal.records.items()}
            for ticks in (cal.tick_min, cal.tick_max, np.rint(cal.zero_ticks)):
                values = bus._normalize({i+1: int(v) for i, v in enumerate(ticks)})
                degrees = np.array([values[i+1] for i in range(5)])
                np.testing.assert_allclose(cal.lerobot_degrees_to_radians(degrees),
                                           cal.ticks_to_radians(ticks), atol=1e-14, rtol=0)
                np.testing.assert_allclose(cal.radians_to_lerobot_degrees(cal.ticks_to_radians(ticks)),
                                           degrees, atol=1e-12, rtol=0)

    def test_synthetic_alignment_fk_jacobian_and_urdf(self):
        # Deliberately synthetic values, not a calibration of either real arm.
        offsets = np.array([.3, -.7, .9, -1.2, .45])
        signs = np.array([-1, 1, -1, 1, -1])
        alignment = json.loads(ALIGNMENT_PATH.read_text())
        with tempfile.TemporaryDirectory(prefix='so101-calibration-test-') as tmp:
            path = Path(tmp)/'alignment.json'
            for arm in ('leader', 'follower'):
                alignment['arms'][arm] = {
                    'confirmed': True,
                    'calibration_sha256': ServoCalibration(arm).fingerprint,
                    'cad_at_zero_rad': dict(zip(MOTOR_NAMES, offsets.tolist())),
                    'cad_sign': dict(zip(MOTOR_NAMES, signs.tolist())),
                }
            path.write_text(json.dumps(alignment))
            base = SO101Kinematics()
            rng = np.random.default_rng(102)
            for arm in ('leader', 'follower'):
                model = CalibratedSO101Kinematics(arm, alignment_path=path)
                for q in [np.zeros(5), *rng.uniform(-.6, .6, (5, 5))]:
                    tcp = [.03, -.01, .07]
                    cad_q = offsets+signs*q
                    np.testing.assert_allclose(model.fk(q, tcp), base.fk(cad_q, tcp), atol=1e-13)
                    np.testing.assert_allclose(model.jacobian(q, tcp),
                                               base.jacobian(cad_q, tcp)*signs, atol=1e-13)
                dest = Path(tmp)/f'{arm}.urdf'
                model.export_arm_urdf(dest)
                root = ET.parse(dest).getroot()
                active = [j for j in root.findall('joint') if j.get('type') != 'fixed']
                self.assertEqual(len(active), 5)
                joints = {j.get('name'): j for j in active}
                for i, name in enumerate(ARM_JOINTS):
                    self.assertEqual(joints[name].get('type'), 'revolute')
                    limit = joints[name].find('limit')
                    self.assertAlmostEqual(float(limit.get('lower')), model.calibration.lower[i])
                    self.assertAlmostEqual(float(limit.get('upper')), model.calibration.upper[i])
                try:
                    importlib.metadata.version('frax')
                except importlib.metadata.PackageNotFoundError:
                    continue
                robot = model.load_frax(tcp)
                for q in (np.zeros(5), np.array([.1, -.3, .4, -.5, .2])):
                    np.testing.assert_allclose(np.asarray(robot.ee_transform(q)), model.fk(q, tcp), atol=2e-8)
                    np.testing.assert_allclose(np.asarray(robot.ee_jacobian(q)), model.jacobian(q, tcp), atol=2e-8)

    def test_stale_alignment_is_rejected(self):
        data = json.loads(ALIGNMENT_PATH.read_text())
        data['arms']['follower']['confirmed'] = True
        data['arms']['follower']['calibration_sha256'] = 'outdated'
        with tempfile.TemporaryDirectory(prefix='so101-stale-alignment-') as tmp:
            path = Path(tmp)/'alignment.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                CalibratedSO101Kinematics('follower', alignment_path=path)

    def test_changed_cad_model_binding_is_rejected(self):
        data = json.loads(ALIGNMENT_PATH.read_text())
        profile = data['arms']['follower']
        profile.update(confirmed=True,
                       calibration_sha256=ServoCalibration('follower').fingerprint,
                       cad_sign={n: 1 for n in MOTOR_NAMES},
                       cad_at_zero_rad={n: 0. for n in MOTOR_NAMES},
                       reference_model_sha256='not-the-current-CAD-model')
        with tempfile.TemporaryDirectory(prefix='so101-model-binding-test-') as tmp:
            path = Path(tmp)/'alignment.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'CAD model archive'):
                CalibratedSO101Kinematics('follower', alignment_path=path)


if __name__ == '__main__':
    unittest.main()
