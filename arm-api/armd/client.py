"""ArmClient: a small Python client for armd. It uses only the standard library.

    from armd.client import ArmClient
    c = ArmClient(arm="follower")
    c.move("gripper", degrees=150, wait=True)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ArmClientError(Exception):
    def __init__(self, status: int, code: str, detail: str):
        super().__init__(f"{status} {code}: {detail}")
        self.status = status
        self.code = code
        self.detail = detail


def _target(position=None, degrees=None, delta=None, delta_degrees=None,
            speed=None, acc=None) -> dict:
    body = {"position": position, "degrees": degrees, "delta": delta,
            "delta_degrees": delta_degrees, "speed": speed, "acc": acc}
    return {k: v for k, v in body.items() if v is not None}


class ArmClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", arm: str = "follower",
                 timeout: float = 10):
        self.base_url = base_url.rstrip("/")
        self.arm = arm
        self.timeout = timeout

    # --- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, body: Any = None,
                 query: dict | None = None, timeout: float | None = None) -> Any:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                err = json.loads(raw)
                code = err.get("error", "http_error")
                detail = err.get("detail", "")
            except (ValueError, AttributeError):
                code, detail = "http_error", raw.decode(errors="replace")
            raise ArmClientError(e.code, code, str(detail)) from None
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise ArmClientError(0, "connection_error", str(reason)) from None
        return json.loads(raw) if raw else None

    def _arm(self, suffix: str = "") -> str:
        return f"/arms/{urllib.parse.quote(self.arm)}{suffix}"

    @staticmethod
    def _q(name: str) -> str:
        return urllib.parse.quote(name)

    # --- API -----------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def arms(self) -> list:
        return self._request("GET", "/arms")

    def state(self, fresh: bool = False) -> dict:
        return self._request("GET", self._arm("/state"), query={"fresh": str(fresh).lower()})

    def joint(self, name: str, fresh: bool = False) -> dict:
        return self._request("GET", self._arm(f"/joints/{self._q(name)}"),
                             query={"fresh": str(fresh).lower()})

    def move(self, joint: str, position: int | None = None, degrees: float | None = None,
             delta: int | None = None, delta_degrees: float | None = None,
             speed: int | None = None, acc: int | None = None, wait: bool = False,
             timeout: float = 5.0, tolerance: int = 10) -> dict:
        body = _target(position, degrees, delta, delta_degrees, speed, acc)
        body.update({"wait": wait, "timeout": timeout, "tolerance": tolerance})
        return self._request("POST", self._arm(f"/joints/{self._q(joint)}/position"), body,
                             timeout=self._wait_timeout(wait, timeout))

    def move_many(self, joints: dict, speed: int | None = None, acc: int | None = None,
                  wait: bool = False, timeout: float = 5.0, tolerance: int = 10) -> dict:
        """joints: {name: {"position": 2048, "speed": 300}} or {name: 2048} (ticks)."""
        spec = {}
        for name, value in joints.items():
            spec[name] = dict(value) if isinstance(value, dict) else {"position": int(value)}
        body: dict[str, Any] = {"joints": spec, "wait": wait, "timeout": timeout,
                                "tolerance": tolerance}
        if speed is not None:
            body["speed"] = speed
        if acc is not None:
            body["acc"] = acc
        return self._request("POST", self._arm("/positions"), body,
                             timeout=self._wait_timeout(wait, timeout))

    def set_speed(self, joint: str, speed: int, acc: int | None = None) -> dict:
        body: dict[str, Any] = {"speed": speed}
        if acc is not None:
            body["acc"] = acc
        return self._request("PUT", self._arm(f"/joints/{self._q(joint)}/speed"), body)

    def spin(self, joint: str, speed: int, acc: int | None = None) -> dict:
        body: dict[str, Any] = {"speed": speed}
        if acc is not None:
            body["acc"] = acc
        return self._request("POST", self._arm(f"/joints/{self._q(joint)}/spin"), body)

    def torque(self, enabled: bool, joints: list[str] | None = None) -> dict:
        body: dict[str, Any] = {"enabled": enabled}
        if joints:
            body["joints"] = list(joints)
        return self._request("POST", self._arm("/torque"), body)

    def stop(self, mode: str = "hold") -> dict:
        return self._request("POST", self._arm("/stop"), {"mode": mode})

    def release(self) -> dict:
        return self._request("POST", self._arm("/release"))

    def acquire(self) -> dict:
        return self._request("POST", self._arm("/acquire"))

    def read_register(self, joint: str, addr: int, size: int = 1) -> dict:
        return self._request("GET", self._arm(f"/joints/{self._q(joint)}/registers/{int(addr)}"),
                             query={"size": size})

    def write_register(self, joint: str, addr: int, value: int, size: int = 1) -> dict:
        return self._request("PUT", self._arm(f"/joints/{self._q(joint)}/registers/{int(addr)}"),
                             {"value": value, "size": size})

    def _wait_timeout(self, wait: bool, timeout: float) -> float:
        # With wait=true the server can take "timeout" seconds to answer.
        return max(self.timeout, timeout + 5.0) if wait else self.timeout
