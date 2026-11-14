"""30 Hz Dual-RealSense D415 + Wrist UVC BabyROS SHM publisher (skips /dev/video0..31 IPU7)."""

from __future__ import annotations

import os
import time

from brickcell.block_inspector.camera_adapters import MulticamHttpSource
from brickcell.block_inspector.runtime_bus import BabyRosRuntimeBus
from brickcell.block_inspector.runtime_nodes import CameraPubNode


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    multicam_url = os.environ.get("MULTICAM_BASE_URL", "http://127.0.0.1:8092")
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    src = MulticamHttpSource(base_url=multicam_url)
    node = CameraPubNode(bus=bus, source=src)
    period_s = 1.0 / 30.0
    try:
        while True:
            t0 = time.monotonic()
            node.publish_step()
            rem = period_s - (time.monotonic() - t0)
            if rem > 0.0:
                time.sleep(rem)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
