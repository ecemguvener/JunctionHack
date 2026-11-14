"""BabyROS <-> armd (127.0.0.1:8765) bridge service node."""

from __future__ import annotations

import os
import time

import numpy as np

from brickcell.block_inspector.armd_bridge import ArmdFollowerBridge
from brickcell.block_inspector.runtime_bus import CANONICAL_TOPICS, BabyRosRuntimeBus


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    armd_url = os.environ.get("ARMD_BASE_URL", "http://127.0.0.1:8765")
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    bridge = ArmdFollowerBridge(base_url=armd_url)

    def _on_command(req: dict) -> dict:
        if req.get("stop"):
            return bridge.emergency_stop_hold()
        q = np.asarray(req["q_enc_rad"], dtype=np.float64)
        grip = float(req.get("gripper_pct", 0.5))
        return bridge.send_command(q, grip, wait=bool(req.get("wait", False)))

    bus.register_server(CANONICAL_TOPICS["arm_command"], _on_command)
    try:
        while True:
            st = bridge.read_follower_state(fresh=False)
            if st is not None:
                bus.publish(
                    CANONICAL_TOPICS["arm_state"],
                    {
                        "q_enc_rad": st.q_enc_rad,
                        "gripper_pct": np.array([st.gripper_pct], dtype=np.float64),
                        "moving": np.array([int(st.moving)], dtype=np.int32),
                    },
                )
            time.sleep(1.0 / 30.0)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
