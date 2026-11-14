"""OpenVINO CPU / Intel Arc XPU / Panther Lake NPU perception & anomaly scoring node.

Subscribes to Dual-RealSense D415 RGB-D + arm state topics, publishes 6D block pose on
`inspection/perception/block_state`, and publishes OpenVINO raw-distance anomaly alerts on
`so101/inspection/anomaly` during Joint-5 CCW Plate-camera scans (never issues robot commands).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from brickcell.block_inspector.openvino_anomaly import ensure_default_plate_openvino_detector
from brickcell.block_inspector.runtime_bus import BabyRosRuntimeBus
from brickcell.block_inspector.runtime_nodes import PerceptionAndAnomalyNode
from brickcell.detector.runtime import OpenVINODetector


def main() -> None:
    shm_bytes = int(os.environ.get("BABYROS_SHM_BYTES", "134217728"))
    model_xml_env = os.environ.get("BRICKCELL_OPENVINO_XML")
    detector = (
        OpenVINODetector(Path(model_xml_env))
        if model_xml_env
        else ensure_default_plate_openvino_detector()
    )
    bus = BabyRosRuntimeBus(shm_pool_bytes=shm_bytes)
    node = PerceptionAndAnomalyNode(bus=bus, detector=detector)
    try:
        while True:
            node.localize_cluster_once()
            time.sleep(0.25)
    finally:
        node.close()
        bus.close()


if __name__ == "__main__":
    main()
