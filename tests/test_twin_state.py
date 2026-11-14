import math

import numpy as np
import pytest

from brickcell.twin.state import (
    PARTS,
    BrickState,
    StationState,
    body_pose_from_mesh_pose,
    matrix_to_quat,
    mesh_pose_from_body_pose,
    quat_to_matrix,
    rot_z,
)


def test_mesh_body_round_trip():
    rng = np.random.default_rng(0)
    for object_id in PARTS:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pos = rng.normal(size=3)
        bp, bq = body_pose_from_mesh_pose(object_id, pos, q)
        mp, mq = mesh_pose_from_body_pose(object_id, bp, bq)
        assert np.allclose(mp, pos)
        assert np.allclose(quat_to_matrix(mq), quat_to_matrix(q))


def test_upright_body_is_grip_height_above_mesh():
    for object_id, part in PARTS.items():
        bp, bq = body_pose_from_mesh_pose(
            object_id, [0.25, 0, 0], matrix_to_quat(rot_z(part.mesh_yaw))
        )
        assert np.allclose(bp, [0.25, 0, part.grip_z])
        assert np.allclose(quat_to_matrix(bq), np.eye(3))


def test_state_json_round_trip_and_validation():
    s = StationState(
        timestamp=1.0, bricks=[BrickState("part-001", "3437", [0, 0, 0], [1, 0, 0, 0], [1, 0, 0])]
    )
    assert StationState.from_json(s.to_json()) == s
    bad = s.to_dict()
    bad["bricks"][0]["object_id"] = "part-999"
    with pytest.raises(ValueError):
        StationState.from_dict(bad)


def test_parts_match_world():
    world = pytest.importorskip("brickcell.simulation.world")
    for object_id, b in world.BRICKS.items():
        p = PARTS[object_id]
        assert (p.part, p.mesh) == (b.part, b.mesh)
        assert math.isclose(p.mesh_yaw, b.yaw) and math.isclose(p.grip_z, b.grip_z)


def test_body_conversion_matches_mujoco_geom():
    mujoco = pytest.importorskip("mujoco")
    from pathlib import Path

    from brickcell.simulation.world import World

    w = World(Path("artifacts/twin-test-world"))
    for object_id in PARTS:
        g = w.model.geom(f"{object_id}-solid").id
        mujoco.mj_forward(w.model, w.data)
        body = w.model.body(object_id).id
        mp, mq = mesh_pose_from_body_pose(object_id, w.data.xpos[body], w.data.xquat[body])
        # the mesh geom frame may be re-centred by MuJoCo; compare via the geom-to-mesh offset
        r_geom = w.data.geom_xmat[g].reshape(3, 3)
        off = w.model.mesh_pos[w.model.geom_dataid[g]]
        mesh_rot_in_geom = quat_to_matrix(w.model.mesh_quat[w.model.geom_dataid[g]])
        r_mesh = r_geom @ mesh_rot_in_geom.T
        p_mesh = w.data.geom_xpos[g] - r_mesh @ off
        assert np.allclose(mp, p_mesh, atol=1e-6), object_id
        assert np.allclose(quat_to_matrix(mq), r_mesh, atol=1e-6), object_id
