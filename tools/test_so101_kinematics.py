"""Read-only numerical checks; no robot controller or device is opened."""
import importlib.metadata
import unittest

import numpy as np

from so101_kinematics import SO101Kinematics


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SO101Kinematics()

    def test_jacobian_against_central_differences(self):
        rng = np.random.default_rng(101)
        for offset in ([0., 0., 0.], [.025, -.012, .075]):
            for q in [np.zeros(5), *rng.uniform(-1, 1, (12, 5))]:
                h = 1e-6
                r0 = self.model.fk(q, offset)[:3, :3]
                columns = []
                for i in range(5):
                    dq = np.zeros(5)
                    dq[i] = h
                    plus, minus = self.model.fk(q+dq, offset), self.model.fk(q-dq, offset)
                    v = (plus[:3, 3]-minus[:3, 3])/(2*h)
                    skew = ((plus[:3, :3]-minus[:3, :3])/(2*h))@r0.T
                    columns.append(np.r_[v, [skew[2, 1], skew[0, 2], skew[1, 0]]])
                np.testing.assert_allclose(self.model.jacobian(q, offset),
                                           np.column_stack(columns), atol=2e-9, rtol=0)

    def test_tcp_offset_changes_wrist_roll_linear_velocity(self):
        at_joint = self.model.jacobian(np.zeros(5))
        at_tcp = self.model.jacobian(np.zeros(5), [.03, 0, .06])
        np.testing.assert_allclose(at_joint[:3, 4], 0, atol=1e-14)
        self.assertGreater(np.linalg.norm(at_tcp[:3, 4]), .01)
        np.testing.assert_allclose(at_joint[3:], at_tcp[3:], atol=1e-14)

    def test_frax_backend(self):
        try:
            importlib.metadata.version('frax')
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('Frax is not installed in this interpreter')
        offset = [.025, -.012, .075]
        robot = self.model.load_frax(offset)
        self.assertEqual(robot.nv, 5)
        for q in (np.zeros(5), np.array([.2, -.4, .6, -.3, .1])):
            np.testing.assert_allclose(np.asarray(robot.ee_transform(q)),
                                       self.model.fk(q, offset), atol=2e-8, rtol=0)
            np.testing.assert_allclose(np.asarray(robot.ee_jacobian(q)),
                                       self.model.jacobian(q, offset), atol=2e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
