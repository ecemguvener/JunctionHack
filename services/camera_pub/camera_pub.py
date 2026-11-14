"""Dual-RealSense D415 + Wrist UVC BabyROS SHM publisher consuming live loopback feeds."""

from __future__ import annotations

import os
import time

from brickcell.block_inspector.camera_adapters import MulticamHttpSource
from brickcell.block_inspector.hardware_cell import LiveBenchHttpFrameSource
from brickcell.block_inspector.runtime_bus import BabyRosRuntimeBus
from brickcell.block_inspector.runtime_nodes import CameraPubNode


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    multicam_url = os.environ.get("MULTICAM_BASE_URL", "")
    live_depth_url = os.environ.get("LIVE_DEPTH_URL", "http://127.0.0.1:8094")
    wrist_http_url = os.environ.get("WRIST_HTTP_URL", "http://127.0.0.1:8091/frame.jpg")
    hz = float(os.environ.get("CAMERA_PUB_HZ", "15.0"))

    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    src = (
        MulticamHttpSource(base_url=multicam_url)
        if multicam_url
        else LiveBenchHttpFrameSource(live_depth_url=live_depth_url, wrist_http_url=wrist_http_url)
    )
    node = CameraPubNode(bus=bus, source=src)
    period_s = 1.0 / max(1.0, hz)
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
