"""Frax 0.2.0 IK + MuJoCo 4.0.1 occlusion raycast & skill service node."""

from __future__ import annotations

import os
import time

from brickcell.block_inspector.runtime_bus import BabyRosRuntimeBus
from brickcell.block_inspector.runtime_nodes import CameraPubNode, FraxMujocoTwinNode
from brickcell.block_inspector.twin import SO101InspectionTwin


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    num_blocks = int(os.environ.get("BRICKCELL_NUM_BLOCKS", "8"))
    show_viewer = os.environ.get("BRICKCELL_SHOW_VIEWER", "0") == "1"
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    twin = SO101InspectionTwin(
        num_blocks=num_blocks,
        show_viewer=show_viewer,
        realtime_pacing=show_viewer,
    )
    cam_pub = (
        CameraPubNode(bus=bus, source=twin)
        if os.environ.get("BRICKCELL_TWIN_PUBLISH_CAMERAS", "1") == "1"
        else None
    )
    node = FraxMujocoTwinNode(bus=bus, twin=twin, camera_pub=cam_pub)
    try:
        while True:
            time.sleep(0.5)
    finally:
        node.close()
        twin.close()
        bus.close()


if __name__ == "__main__":
    main()
