"""Real-time BabyROS node tracking green (`good`) & red/pink (`defective`) bowl bounding circles.

Subscribes to `so101/camera/realsense_1/frame` (Top D415 RGB) and publishes live world XY +
image bounding-circle centers on `so101/perception/bowl_circles` (never issues robot commands).
"""

from __future__ import annotations

import os
import time

from brickcell.block_inspector.real2sim import real_intel_camera_poses_world
from brickcell.block_inspector.runtime_bus import BabyRosRuntimeBus
from brickcell.block_inspector.runtime_nodes import BowlCircleTrackerNode


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    hz = float(os.environ.get("BOWL_TRACKER_HZ", "15.0"))
    R_top, p_top, K_top = real_intel_camera_poses_world()["top"]
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    node = BowlCircleTrackerNode(bus=bus, R_top=R_top, p_top=p_top, K_top=K_top)
    period_s = 1.0 / max(1.0, hz)
    try:
        while True:
            time.sleep(period_s)
    finally:
        node.close()
        bus.close()


if __name__ == "__main__":
    main()
