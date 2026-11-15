"""Zero-copy babyros / Zenoh SHM inter-node runtime bus + deterministic loopback bus.

Matches ARCHITECTURE.md §6.1:
- Calls `babyros.configure(...)` BEFORE any node creation so the 128 MiB Zenoh shared-memory
  pool is active on localhost / Wendy `shared-ipc` containers.
- Enforces BabyROS's strict no-leading-'/' topic invariant across all canonical topics.
- Disables Arrow IPC LZ4 compression (`compression=None`) on localhost SHM `NDDC` array dicts.
- Provides `InProcessRuntimeBus` with identical topic/schema semantics for deterministic
  sandboxed unit tests where OS sockets are disabled.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

CANONICAL_TOPICS = {
    "realsense_1_frame": "so101/camera/realsense_1/frame",
    "realsense_2_frame": "so101/camera/realsense_2/frame",
    "wrist_frame": "so101/camera/wrist/frame",
    "arm_state": "so101/arm/state",
    "block_state": "inspection/perception/block_state",
    "bowl_circles": "so101/perception/bowl_circles",
    "inspection_anomaly": "so101/inspection/anomaly",
    "symbolic_state": "so101/symbolic/state",
    "skill_execute": "so101/skill/execute",
    "arm_command": "so101/arm/command",
}

DEFAULT_SHM_POOL_BYTES = 134_217_728  # 128 MiB


def validate_babyros_topic(topic: str) -> str:
    """Validate BabyROS topic naming rules (non-empty, strictly no leading '/')."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Invalid topic: topic must be a non-empty string.")
    if topic.startswith("/"):
        raise ValueError("Topic names should not start with '/'.")
    return topic


def pack_rgbd_frame_nddc(
    rgb: np.ndarray,
    timestamp_ns: int,
    depth_mm: np.ndarray | None = None,
    intrinsics_k: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pack RGB (+ optional aligned uint16 depth_mm) into a BabyROS NDDC dict of arrays."""
    payload: dict[str, np.ndarray] = {
        "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
        "timestamp_ns": np.array([int(timestamp_ns)], dtype=np.int64),
    }
    if depth_mm is not None:
        payload["depth_mm"] = np.ascontiguousarray(depth_mm, dtype=np.uint16)
    if intrinsics_k is not None:
        payload["K"] = np.ascontiguousarray(intrinsics_k, dtype=np.float64)
    return payload


class InProcessRuntimeBus:
    """In-memory pub/sub + RPC bus mirroring BabyROS topic & durability contracts."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Any], None]]] = {}
        self._latched: dict[str, Any] = {}
        self._servers: dict[str, Callable[[Any], Any]] = {}

    def publish(self, topic: str, data: Any, transient_local: bool = False) -> None:
        validate_babyros_topic(topic)
        msg = copy.deepcopy(data) if isinstance(data, dict) else data
        if transient_local:
            self._latched[topic] = msg
        for cb in list(self._subs.get(topic, ())):
            cb(msg)

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Any], None],
        transient_local: bool = False,
    ) -> Callable[[], None]:
        validate_babyros_topic(topic)
        self._subs.setdefault(topic, []).append(callback)
        if transient_local and topic in self._latched:
            callback(self._latched[topic])

        def _unsubscribe() -> None:
            if topic in self._subs and callback in self._subs[topic]:
                self._subs[topic].remove(callback)

        return _unsubscribe

    def register_server(self, service_topic: str, handler: Callable[[Any], Any]) -> None:
        validate_babyros_topic(service_topic)
        self._servers[service_topic] = handler

    def call_service(self, service_topic: str, request: Any) -> Any:
        validate_babyros_topic(service_topic)
        if service_topic not in self._servers:
            raise KeyError(f"No server registered for '{service_topic}'")
        return self._servers[service_topic](copy.deepcopy(request))

    def close(self) -> None:
        self._subs.clear()
        self._latched.clear()
        self._servers.clear()


class BabyRosRuntimeBus:
    """Zero-copy Zenoh shared-memory bus backed by babyros==0.1.10."""

    def __init__(
        self,
        shm_pool_bytes: int = DEFAULT_SHM_POOL_BYTES,
        connect_endpoints: list[str] | None = None,
        listen_endpoints_peer: list[str] | None = None,
    ) -> None:
        import babyros
        from babyros.node import Durability, SessionManager

        self._Durability = Durability
        self._SessionManager = SessionManager
        self._nodes: list[Any] = []
        self._pubs: dict[tuple[str, bool], Any] = {}

        if SessionManager._session is None:
            babyros.configure(
                mode="peer",
                connect_endpoints=connect_endpoints,
                listen_endpoints_peer=listen_endpoints_peer,
                scouting_multicast_enabled=False,
                transport_shared_memory_enabled=True,
                transport_shared_memory_pool_size=int(shm_pool_bytes),
                transport_shared_memory_message_size_threshold=4096,
            )

    def publish(self, topic: str, data: Any, transient_local: bool = False) -> None:
        from babyros.node import Publisher

        validate_babyros_topic(topic)
        key = (topic, bool(transient_local))
        pub = self._pubs.get(key)
        if pub is None:
            dur = self._Durability.TRANSIENT_LOCAL if transient_local else self._Durability.VOLATILE
            pub = Publisher(topic=topic, compression=None, durability=dur)
            self._pubs[key] = pub
            self._nodes.append(pub)
        pub.publish(data)

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Any], None],
        transient_local: bool = False,
    ) -> Callable[[], None]:
        from babyros.node import Subscriber

        validate_babyros_topic(topic)
        dur = self._Durability.TRANSIENT_LOCAL if transient_local else self._Durability.VOLATILE
        sub = Subscriber(topic=topic, callback=callback, durability=dur)
        self._nodes.append(sub)
        return sub.delete

    def register_server(self, service_topic: str, handler: Callable[[Any], Any]) -> Any:
        from babyros.node import Server

        validate_babyros_topic(service_topic)
        srv = Server(service_topic, handler)
        self._nodes.append(srv)
        return srv

    def call_service(self, service_topic: str, request: Any, timeout: float = 15.0) -> Any:
        from babyros.node import Client

        validate_babyros_topic(service_topic)
        cli = Client(service_topic, timeout=timeout)
        try:
            return cli.request(request)
        finally:
            cli.delete()

    def close(self) -> None:
        self._pubs.clear()
        for node in list(self._nodes):
            try:
                node.delete()
            except Exception:
                pass
        self._nodes.clear()
        if self._SessionManager._session is not None:
            self._SessionManager.delete(force=True)
