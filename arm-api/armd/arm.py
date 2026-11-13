"""Arm: one SO-101 arm on one bus. Safety rules, clamps, and the poller.

Lock order: bus.lock first, then self._lock. Never get bus.lock while you
hold self._lock. All multi-step bus sequences hold bus.lock for the full
sequence, so the poller cannot read between the steps.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any

from .config import ArmConfig
from .types import (
    ACC_MAX,
    EEPROM_END,
    MODE_POSITION,
    MODE_WHEEL,
    POSITION_MAX,
    POSITION_MIN,
    SPEED_MAX,
    TICKS_PER_REV,
    ArmApiError,
    BusError,
    BusProtocol,
    BusReleased,
    InvalidRequest,
    JointFault,
    MotionDisabled,
    NotFound,
    ServoError,
    ServoReading,
    decode_errors,
    ticks_to_degrees,
)

log = logging.getLogger("armd.arm")

TARGET_KEYS = ("position", "degrees", "delta", "delta_degrees")
LOCK_REGISTER = 55
# The poller marks the arm offline after this many failed polls in a row.
MAX_POLL_FAILURES = 3
MAX_WAIT_TIMEOUT = 120.0


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Arm:
    def __init__(self, name: str, cfg: ArmConfig, bus: BusProtocol, *,
                 poll_hz: float = 30.0, retry_interval: float = 2.0, admin: bool = False):
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.poll_hz = float(poll_hz)
        self.retry_interval = float(retry_interval)
        self.admin = admin

        self._ids = [j.id for j in cfg.joints.values()]
        self._by_id = {j.id: j.name for j in cfg.joints.values()}

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._readings: dict[int, ServoReading] = {}
        self._timestamp: float | None = None   # unix time of the last good read
        self._seq = 0                          # count of good reads
        self._status = "offline"
        self._error: str | None = "not started"
        self._released = False
        self._failures = 0
        self._next_retry = 0.0

        # Data that is read from the servos at connect.
        self._eeprom: dict[str, tuple[int, int]] = {}
        self._modes: dict[int, int] = {}
        self._info_retry: dict[int, float] = {}

        # Runtime defaults (PUT .../speed changes them, not saved to file).
        self._speed = {n: min(j.default_speed or cfg.default_speed, cfg.max_speed)
                       for n, j in cfg.joints.items()}
        self._acc = {n: (j.default_acc if j.default_acc is not None else cfg.default_acc)
                     for n, j in cfg.joints.items()}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- life cycle ----------------------------------------------------------

    def start(self) -> None:
        """Connect once now (no exception if it fails), then start the poller."""
        self._connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"poller-{self.name}", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self.bus.lock:
            if self._released:
                return
            try:
                if self.bus.is_open and self.cfg.on_shutdown == "limp":
                    log.info("arm %s: on_shutdown=limp, torque off", self.name)
                    self.bus.set_torque(list(self._ids), False)
            except Exception as e:  # noqa: BLE001 - shutdown must continue
                log.warning("arm %s: torque off at shutdown failed: %s", self.name, e)
            try:
                self.bus.close()
            except Exception as e:  # noqa: BLE001
                log.warning("arm %s: close failed: %s", self.name, e)
            with self._cond:
                self._status = "offline"
                self._error = "shut down"
                self._readings = {}
                self._cond.notify_all()

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    # --- connect and poll ----------------------------------------------------

    def _connect(self) -> None:
        with self.bus.lock:
            if self._released:
                return
            try:
                if not self.bus.is_open:
                    self.bus.open()
                self._modes.clear()
                readings = self.bus.read_states(list(self._ids))
                self._fill_info(readings, force=True)
            except Exception as e:  # noqa: BLE001 - any failure = offline
                self._mark_offline(e)
                return
            self._failures = 0
            self._store(readings)
            log.info("arm %s: online on %s (%d/%d servos)", self.name, self.bus.port,
                     len(readings), len(self._ids))

    def _mark_offline(self, err: Exception | str) -> None:
        """Close the port and retry later. The caller holds bus.lock."""
        try:
            self.bus.close()
        except Exception:  # noqa: BLE001
            pass
        with self._cond:
            was = self._status
            if not self._released:
                self._status = "offline"
            self._readings = {}
            self._error = str(err) or type(err).__name__
            self._next_retry = time.monotonic() + self.retry_interval
            self._cond.notify_all()
        if was != "offline":
            log.warning("arm %s: offline: %s", self.name, self._error)

    def _store(self, readings: dict[int, ServoReading]) -> None:
        with self._cond:
            self._readings = dict(readings)
            self._timestamp = time.time()
            self._seq += 1
            if not self._released:
                self._status = "online"
                self._error = None
            self._cond.notify_all()

    def _fill_info(self, readings: dict[int, ServoReading], force: bool = False) -> None:
        """Read EEPROM limits and mode of servos that answered and are not known yet."""
        now = time.monotonic()
        for sid in readings:
            name = self._by_id.get(sid)
            if name is None:
                continue
            j = self.cfg.joints[name]
            need_limits = (j.min is None or j.max is None) and name not in self._eeprom
            need_mode = sid not in self._modes
            if not (need_limits or need_mode):
                continue
            if not force and now < self._info_retry.get(sid, 0.0):
                continue
            try:
                if need_limits:
                    lo, hi = self.bus.read_limits(sid)
                    if not (POSITION_MIN <= lo < hi <= POSITION_MAX):
                        lo, hi = POSITION_MIN, POSITION_MAX
                    self._eeprom[name] = (lo, hi)
                if need_mode:
                    self._modes[sid] = self.bus.get_mode(sid)
            except BusError:
                raise
            except Exception as e:  # noqa: BLE001
                self._info_retry[sid] = now + self.retry_interval
                log.warning("arm %s: cannot read limits/mode of id %d: %s", self.name, sid, e)

    def _poll_once(self) -> None:
        with self.bus.lock:
            if self._released or self._status != "online":
                return
            try:
                readings = self.bus.read_states(list(self._ids))
                self._fill_info(readings)
            except Exception as e:  # noqa: BLE001
                self._failures += 1
                if not self.bus.is_open or self._failures >= MAX_POLL_FAILURES:
                    self._mark_offline(e)
                else:
                    log.debug("arm %s: poll failed (%d): %s", self.name, self._failures, e)
                return
            self._failures = 0
            self._store(readings)

    def _run(self) -> None:
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self._released:
                    self._stop.wait(0.05)
                    continue
                if self._status != "online":
                    if t0 >= self._next_retry:
                        self._connect()
                    else:
                        self._stop.wait(min(0.05, self._next_retry - t0))
                        continue
                else:
                    self._poll_once()
            except Exception:  # noqa: BLE001 - the poller must not stop
                log.exception("arm %s: poller error", self.name)
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))

    def _read_now(self) -> dict[int, ServoReading]:
        """Read all servos now and update the cache. The caller holds bus.lock."""
        readings = self.bus.read_states(list(self._ids))
        self._store(readings)
        return readings

    # --- checks --------------------------------------------------------------

    def _check_not_released(self) -> None:
        if self._released:
            raise BusReleased(f"arm {self.name} is released. POST /arms/{self.name}/acquire first.")

    def _check_connected(self) -> None:
        self._check_not_released()
        if self._status != "online" or not self.bus.is_open:
            raise BusError(f"arm {self.name} is offline: {self._error}")

    def _check_motion(self) -> None:
        if not self.cfg.allow_motion:
            raise MotionDisabled(f"arm {self.name} is read-only (allow_motion is false)")

    def _joint(self, name: str):
        j = self.cfg.joints.get(name)
        if j is None:
            raise NotFound(f"arm {self.name} has no joint {name!r}")
        return j

    def _speed_value(self, v: Any) -> int:
        """Validate a position-mode speed and clamp it to max_speed."""
        if not _is_int(v):
            raise InvalidRequest(f"speed must be an integer, not {v!r}")
        if v == 0:
            raise InvalidRequest("speed 0 is not allowed (0 means maximum speed on the servo)")
        if not 1 <= v <= SPEED_MAX:
            raise InvalidRequest(f"speed must be 1-{SPEED_MAX}, not {v}")
        return min(v, self.cfg.max_speed)

    @staticmethod
    def _acc_value(v: Any) -> int:
        if not _is_int(v) or not 0 <= v <= ACC_MAX:
            raise InvalidRequest(f"acc must be an integer 0-{ACC_MAX}, not {v!r}")
        return v

    def _errors(self, r: ServoReading) -> list[str]:
        return list(r.errors) if r.errors else decode_errors(r.status)

    def _check_faults(self, name: str, r: ServoReading) -> None:
        blocking = [e for e in self._errors(r) if e in self.cfg.block_on_errors]
        if blocking:
            raise JointFault(f"joint {name} reports {', '.join(blocking)}. "
                             f"Clear the error (for example, torque off) before you move it.")

    def limits(self, name: str) -> tuple[int | None, int | None]:
        j = self._joint(name)
        eeprom = self._eeprom.get(name, (None, None))
        lo = j.min if j.min is not None else eeprom[0]
        hi = j.max if j.max is not None else eeprom[1]
        return lo, hi

    def _patch_torque(self, ids: list[int], enabled: bool) -> None:
        with self._lock:
            for sid in ids:
                r = self._readings.get(sid)
                if r is not None:
                    self._readings[sid] = dataclasses.replace(r, torque_enabled=enabled)

    # --- read-only views -----------------------------------------------------

    def info(self) -> dict:
        return {
            "name": self.name,
            "port": self.cfg.port,
            "status": self.status,
            "allow_motion": self.cfg.allow_motion,
            "allow_spin": self.cfg.allow_spin,
            "max_speed": self.cfg.max_speed,
            "default_speed": self.cfg.default_speed,
            "default_acc": self.cfg.default_acc,
            "joints": [{"name": n, "id": j.id, "min": self.limits(n)[0], "max": self.limits(n)[1]}
                       for n, j in self.cfg.joints.items()],
        }

    def health(self) -> dict:
        with self._lock:
            return {"status": self._status, "port": self.cfg.port,
                    "last_poll": self._timestamp, "error": self._error}

    def _joint_dict(self, name: str, r: ServoReading | None) -> dict:
        j = self.cfg.joints[name]
        lo, hi = self.limits(name)
        d: dict[str, Any] = {"name": name, "id": j.id}
        if r is None:
            for k in ("position", "degrees", "goal_position", "speed", "goal_speed",
                      "acceleration", "load", "voltage", "temperature", "current_ma",
                      "moving", "torque_enabled", "mode"):
                d[k] = None
            d["errors"] = []
            d["healthy"] = False
        else:
            errors = self._errors(r)
            d.update({
                "position": r.position,
                "degrees": round(ticks_to_degrees(r.position), 3),
                "goal_position": r.goal_position,
                "speed": r.speed,
                "goal_speed": r.goal_speed,
                "acceleration": r.acceleration,
                "load": r.load,
                "voltage": r.voltage,
                "temperature": r.temperature,
                "current_ma": r.current_ma,
                "moving": r.moving,
                "torque_enabled": r.torque_enabled,
                "mode": self._modes.get(j.id),
                "errors": errors,
                "healthy": not errors,
            })
        d.update({"min": lo, "max": hi, "default_speed": self._speed[name],
                  "default_acc": self._acc[name], "online": r is not None})
        return d

    def state(self, fresh: bool = False) -> dict:
        if fresh:
            with self.bus.lock:
                self._check_connected()
                self._read_now()
        with self._lock:
            ts = self._timestamp if self._readings else None
            return {
                "arm": self.name,
                "status": self._status,
                "timestamp": ts,
                "age_ms": round((time.time() - ts) * 1000.0, 1) if ts is not None else None,
                "joints": {n: self._joint_dict(n, self._readings.get(j.id))
                           for n, j in self.cfg.joints.items()},
            }

    def joint_state(self, name: str, fresh: bool = False) -> dict:
        j = self._joint(name)
        if fresh:
            with self.bus.lock:
                self._check_connected()
                self._read_now()
        with self._lock:
            return self._joint_dict(name, self._readings.get(j.id))

    # --- moves ---------------------------------------------------------------

    def move(self, joint: str, position: int | None = None, degrees: float | None = None,
             delta: int | None = None, delta_degrees: float | None = None,
             speed: int | None = None, acc: int | None = None, wait: bool = False,
             timeout: float = 5.0, tolerance: int = 10) -> dict:
        spec = {"position": position, "degrees": degrees, "delta": delta,
                "delta_degrees": delta_degrees, "speed": speed, "acc": acc}
        res = self._move({joint: spec}, None, None, wait, timeout, tolerance)
        t = res["targets"][joint]
        final = res["final_positions"]
        return {"joint": joint, "target": t["target"], "clamped": t["clamped"],
                "speed": t["speed"], "acc": t["acc"], "reached": res["reached"],
                "final_position": final.get(joint) if final is not None else None}

    def move_many(self, joints: dict[str, dict], speed: int | None = None, acc: int | None = None,
                  wait: bool = False, timeout: float = 5.0, tolerance: int = 10) -> dict:
        return self._move(joints, speed, acc, wait, timeout, tolerance)

    def _move(self, specs: dict[str, dict], speed: Any, acc: Any, wait: bool,
              timeout: Any, tolerance: Any) -> dict:
        self._check_not_released()
        self._check_motion()
        if not specs:
            raise InvalidRequest("give at least one joint")
        if not _is_num(timeout) or not 0 < timeout <= MAX_WAIT_TIMEOUT:
            raise InvalidRequest(f"timeout must be > 0 and <= {MAX_WAIT_TIMEOUT} s")
        if not _is_int(tolerance) or not 0 <= tolerance <= POSITION_MAX:
            raise InvalidRequest("tolerance must be an integer 0-4095")
        top_speed = self._speed_value(speed) if speed is not None else None
        top_acc = self._acc_value(acc) if acc is not None else None

        # Parse and validate all joints before the first bus call.
        plan: dict[str, tuple[int, str, Any, int, int]] = {}
        for name, spec in specs.items():
            j = self._joint(name)
            spec = dict(spec or {})
            unknown = set(spec) - set(TARGET_KEYS) - {"speed", "acc"}
            if unknown:
                raise InvalidRequest(f"joint {name}: unknown fields {sorted(unknown)}")
            given = [k for k in TARGET_KEYS if spec.get(k) is not None]
            if len(given) != 1:
                raise InvalidRequest(f"joint {name}: give exactly one of {', '.join(TARGET_KEYS)}")
            kind = given[0]
            value = spec[kind]
            if kind in ("position", "delta") and not _is_int(value):
                raise InvalidRequest(f"joint {name}: {kind} must be an integer")
            if kind in ("degrees", "delta_degrees") and not _is_num(value):
                raise InvalidRequest(f"joint {name}: {kind} must be a number")
            sp = (self._speed_value(spec["speed"]) if spec.get("speed") is not None
                  else top_speed if top_speed is not None else self._speed[name])
            ac = (self._acc_value(spec["acc"]) if spec.get("acc") is not None
                  else top_acc if top_acc is not None else self._acc[name])
            plan[name] = (j.id, kind, value, min(sp, self.cfg.max_speed), ac)

        result: dict[str, dict] = {}
        with self.bus.lock:
            self._check_connected()
            if any(p[1] in ("delta", "delta_degrees") for p in plan.values()):
                # A delta uses the latest present position: read it now.
                self._read_now()
            with self._lock:
                readings = dict(self._readings)
            targets: dict[int, tuple[int, int, int]] = {}
            for name, (sid, kind, value, sp, ac) in plan.items():
                r = readings.get(sid)
                if r is None:
                    raise ServoError(f"joint {name} (id {sid}) did not answer at the last poll")
                self._check_faults(name, r)
                if kind == "position":
                    raw = int(value)
                elif kind == "degrees":
                    raw = round(value * TICKS_PER_REV / 360.0)
                elif kind == "delta":
                    raw = r.position + int(value)
                else:
                    raw = r.position + round(value * TICKS_PER_REV / 360.0)
                lo, hi = self.limits(name)
                lo = POSITION_MIN if lo is None else lo
                hi = POSITION_MAX if hi is None else hi
                target = max(lo, min(hi, raw))
                targets[sid] = (target, sp, ac)
                result[name] = {"target": target, "clamped": target != raw, "speed": sp, "acc": ac}

            # Joints in another mode: torque off, set position mode. Then write
            # the goals, then torque on. So the servo goes directly to the new goal.
            ids = list(targets)
            change_mode = []
            for sid in ids:
                if sid not in self._modes:
                    self._modes[sid] = self.bus.get_mode(sid)
                if self._modes[sid] != MODE_POSITION:
                    change_mode.append(sid)
            torque_on = {sid for sid in ids if readings[sid].torque_enabled}
            off_first = [sid for sid in change_mode if sid in torque_on]
            if off_first:
                self.bus.set_torque(off_first, False)
                torque_on -= set(off_first)
            for sid in change_mode:
                self.bus.set_mode(sid, MODE_POSITION)
                self._modes[sid] = MODE_POSITION
            self.bus.write_positions(targets)
            need_on = [sid for sid in ids if sid not in torque_on]
            if need_on:
                self.bus.set_torque(need_on, True)
                self._patch_torque(need_on, True)
            with self._lock:
                write_seq = self._seq

        reached = None
        finals = None
        if wait:
            reached, finals_by_id = self._wait({sid: t[0] for sid, t in targets.items()},
                                               write_seq, float(timeout), int(tolerance))
            finals = {name: finals_by_id.get(plan[name][0]) for name in plan}
        return {"targets": result, "reached": reached, "final_positions": finals}

    def _wait(self, targets: dict[int, int], write_seq: int, timeout: float,
              tolerance: int) -> tuple[bool, dict[int, int]]:
        """Wait until all joints are within tolerance of the target, or timeout."""
        deadline = time.monotonic() + timeout
        poller = self._thread is not None and self._thread.is_alive()
        reached = False
        while True:
            if not poller:
                try:
                    with self.bus.lock:
                        self._check_connected()
                        self._read_now()
                except ArmApiError:
                    pass
            with self._cond:
                if self._seq > write_seq:
                    rs = self._readings
                    if all(sid in rs and abs(rs[sid].position - t) <= tolerance
                           for sid, t in targets.items()):
                        reached = True
                if reached or self._status != "online":
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if poller:
                    self._cond.wait(min(remaining, 0.1))
            if not poller:
                time.sleep(min(max(remaining, 0.0), 0.02))
        with self._lock:
            finals = {sid: self._readings[sid].position for sid in targets if sid in self._readings}
        return reached, finals

    def set_speed(self, joint: str, speed: Any, acc: Any = None) -> dict:
        self._check_not_released()
        self._joint(joint)
        sp = self._speed_value(speed)
        ac = self._acc_value(acc) if acc is not None else None
        with self._lock:
            self._speed[joint] = sp
            if ac is not None:
                self._acc[joint] = ac
            return {"joint": joint, "default_speed": self._speed[joint],
                    "default_acc": self._acc[joint]}

    def spin(self, joint: str, speed: Any, acc: Any = None) -> dict:
        self._check_not_released()
        j = self._joint(joint)
        self._check_motion()
        if not self.cfg.allow_spin:
            raise MotionDisabled(f"arm {self.name} does not allow spin (allow_spin is false)")
        if not _is_int(speed) or not -SPEED_MAX <= speed <= SPEED_MAX:
            raise InvalidRequest(f"spin speed must be an integer -{SPEED_MAX}..{SPEED_MAX}")
        sp = max(-self.cfg.max_speed, min(self.cfg.max_speed, speed))
        ac = self._acc_value(acc) if acc is not None else self._acc[joint]
        sid = j.id
        with self.bus.lock:
            self._check_connected()
            with self._lock:
                r = self._readings.get(sid)
            if r is None:
                raise ServoError(f"joint {joint} (id {sid}) did not answer at the last poll")
            if sp != 0:
                self._check_faults(joint, r)
            if sid not in self._modes:
                self._modes[sid] = self.bus.get_mode(sid)
            if self._modes[sid] == MODE_WHEEL and r.torque_enabled:
                self.bus.spin(sid, sp, ac)
            else:
                # Write mode and speed with torque off. Then torque on. So the
                # joint does not move with an old goal for a short time.
                if r.torque_enabled:
                    self.bus.set_torque([sid], False)
                if self._modes[sid] != MODE_WHEEL:
                    self.bus.set_mode(sid, MODE_WHEEL)
                    self._modes[sid] = MODE_WHEEL
                self.bus.spin(sid, sp, ac)
                self.bus.set_torque([sid], True)
                self._patch_torque([sid], True)
        return {"joint": joint, "speed": sp, "acc": ac, "mode": MODE_WHEEL}

    def torque(self, enabled: bool, joints: list[str] | None = None) -> dict:
        self._check_not_released()
        names = list(joints) if joints else list(self.cfg.joints)
        ids = [self._joint(n).id for n in names]
        if enabled:
            self._check_motion()
        with self.bus.lock:
            self._check_connected()
            if enabled:
                # Set goal = present before torque on, so the joint does not jump.
                with self._lock:
                    off = [sid for sid in ids
                           if not (sid in self._readings and self._readings[sid].torque_enabled)]
                for sid in off:
                    if self._modes.get(sid) == MODE_WHEEL:
                        self.bus.spin(sid, 0, self._acc[self._by_id[sid]])
                pos_off = [sid for sid in off if self._modes.get(sid) != MODE_WHEEL]
                if pos_off:
                    self.bus.hold(pos_off)
            self.bus.set_torque(ids, bool(enabled))
            self._patch_torque(ids, bool(enabled))
        return {"torque": {n: bool(enabled) for n in names}}

    def stop(self, mode: str = "hold") -> dict:
        self._check_not_released()
        if mode not in ("hold", "limp"):
            raise InvalidRequest("mode must be 'hold' or 'limp'")
        ids = list(self._ids)
        with self.bus.lock:
            self._check_connected()
            if mode == "hold":
                for sid in ids:
                    if self._modes.get(sid) == MODE_WHEEL:
                        try:
                            self.bus.spin(sid, 0, self._acc[self._by_id[sid]])
                        except ServoError as e:
                            log.warning("arm %s: stop spin id %d: %s", self.name, sid, e)
                positions = self._hold_best_effort(ids)
            else:
                try:
                    self.bus.set_torque(ids, False)
                except ServoError:
                    for sid in ids:
                        try:
                            self.bus.set_torque([sid], False)
                        except ServoError as e:
                            log.warning("arm %s: limp id %d: %s", self.name, sid, e)
                self._patch_torque(ids, False)
                with self._lock:
                    positions = {sid: r.position for sid, r in self._readings.items()}
        return {"mode": mode, "positions": {self._by_id[sid]: int(p) for sid, p in positions.items()
                                            if sid in self._by_id}}

    def _hold_best_effort(self, ids: list[int]) -> dict[int, int]:
        try:
            return dict(self.bus.hold(ids))
        except ServoError:
            out: dict[int, int] = {}
            for sid in ids:
                try:
                    out.update(self.bus.hold([sid]))
                except ServoError as e:
                    log.warning("arm %s: hold id %d: %s", self.name, sid, e)
            if not out:
                raise
            return out

    # --- release / acquire ---------------------------------------------------

    def release(self) -> dict:
        with self.bus.lock:
            self._released = True
            try:
                self.bus.close()
            except Exception as e:  # noqa: BLE001
                log.warning("arm %s: close failed: %s", self.name, e)
            with self._cond:
                self._status = "released"
                self._readings = {}
                self._error = None
                self._cond.notify_all()
        log.info("arm %s: released %s", self.name, self.cfg.port)
        return {"arm": self.name, "status": "released"}

    def acquire(self) -> dict:
        with self.bus.lock:
            was_released = self._released
            self._released = False
            if was_released or self._status != "online":
                with self._cond:
                    self._status = "offline"
                self._connect()
        return {"arm": self.name, "status": self.status}

    # --- raw registers (admin) -----------------------------------------------

    def _check_admin(self) -> None:
        if not self.admin:
            raise MotionDisabled("raw register access is off. Set server.admin or ARMD_ADMIN=1.")

    @staticmethod
    def _check_reg(addr: Any, size: Any) -> None:
        if not _is_int(addr) or not 0 <= addr <= 255:
            raise InvalidRequest("addr must be an integer 0-255")
        if size not in (1, 2) or isinstance(size, bool):
            raise InvalidRequest("size must be 1 or 2")

    def read_register(self, joint: str, addr: int, size: int = 1) -> dict:
        self._check_admin()
        self._check_not_released()
        j = self._joint(joint)
        self._check_reg(addr, size)
        with self.bus.lock:
            self._check_connected()
            value = self.bus.read_register(j.id, addr, size)
        return {"addr": addr, "size": size, "value": value}

    def write_register(self, joint: str, addr: int, value: int, size: int = 1) -> dict:
        self._check_admin()
        self._check_not_released()
        j = self._joint(joint)
        self._check_reg(addr, size)
        last = addr + size - 1
        if addr < EEPROM_END or addr <= LOCK_REGISTER <= last:
            raise MotionDisabled(f"write to address {addr} is refused "
                                 f"(EEPROM is below {EEPROM_END}, {LOCK_REGISTER} is the lock)")
        if not _is_int(value) or not 0 <= value < (1 << (8 * size)):
            raise InvalidRequest(f"value must be an integer 0-{(1 << (8 * size)) - 1}")
        with self.bus.lock:
            self._check_connected()
            self.bus.write_register(j.id, addr, size, value)
            # The write can change mode or torque: read them again.
            self._modes.pop(j.id, None)
            if addr == 40:
                self._patch_torque([j.id], bool(value & 0xFF))
        return {"addr": addr, "size": size, "value": value}
