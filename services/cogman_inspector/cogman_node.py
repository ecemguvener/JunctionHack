"""CogMan bilevel inspection executive & port 8095 HUD service node."""

from __future__ import annotations

import os
import time

from brickcell.block_inspector.runtime_bus import BabyRosRuntimeBus
from brickcell.block_inspector.runtime_nodes import CogManInspectorNode


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    hud_port = int(os.environ.get("COGMAN_HUD_PORT", "8095"))
    cam_ui_url = os.environ.get("CAM_UI_URL")
    instruction = os.environ.get(
        "BRICKCELL_INSTRUCTION",
        "pick and classify all 8 blocks from source plate",
    )
    run_once = os.environ.get("BRICKCELL_RUN_ONCE", "0") == "1"
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    node = CogManInspectorNode(bus=bus, hud_port=hud_port, cam_ui_url=cam_ui_url)
    try:
        node.run_episode(instruction=instruction)
        while not run_once:
            time.sleep(1.0)
    finally:
        node.close()
        bus.close()


if __name__ == "__main__":
    main()
