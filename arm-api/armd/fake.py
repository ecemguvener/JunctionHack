"""FakeBus: a simulated bus with STS3215 servos. No hardware.

Positions move to the goal at the goal speed in real time, so wait=true and
"moving" work. Tests can change the state directly, for example:

    fake.servos[6].status = 1 << 5      # overload error on servo 6
    fake.servos[3].online = False       # servo 3 does not answer
    fake.port_present = False           # open() raises BusError
    fake.fail = True                    # all calls raise BusError (link fails)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from .types import (
    MODE_POSITION,
    MODE_WHEEL,
    MODEL_STS3215,
    POSITION_MAX,
    POSITION_MIN,
    SPEED_MAX,
    TICKS_PER_REV,
    BusError,
    ServoError,
    ServoReading,
    decode_errors,
)


@dataclass
class FakeServo:
    id: int
    position: float = 2048.0
    goal_position: int = 2048
    goal_speed: int = 0            # 0 = maximum speed (as on the real servo)
    acc: int = 0
    wheel_speed: int = 0           # signed ticks/s in wheel mode
    torque_enabled: bool = False
    mode: int = MODE_POSITION
    status: int = 0
    voltage: float = 12.0
    temperature: int = 30
    load: float = 0.0
    current_ma: float = 0.0
    min_limit: int = POSITION_MIN
    max_limit: int = POSITION_MAX
    lock_flag: int = 1
    online: bool = True
    registers: dict[int, int] = field(default_factory=dict)
    velocity: float = 0.0          # present speed, signed ticks/s
    last_update: float = field(default_factory=time.monotonic)

    def update(self, now: float) -> None:
        """Move the simulated servo forward in time to now."""
        dt = max(0.0, now - self.last_update)
        self.last_update = now
        self.velocity = 0.0
        if not self.torque_enabled or dt == 0.0:
            return
        if self.mode == MODE_WHEEL:
            if self.wheel_speed:
                self.position = (self.position + self.wheel_speed * dt) % TICKS_PER_REV
                self.velocity = float(self.wheel_speed)
            return
        if self.mode != MODE_POSITION:
            return
        error = self.goal_position - self.position
        if abs(error) < 0.5:
            self.position = float(self.goal_position)
            return
        speed = self.goal_speed if self.goal_speed > 0 else SPEED_MAX
        step = speed * dt
        if step >= abs(error):
            self.position = float(self.goal_position)
        else:
            self.position += step if error > 0 else -step
        self.velocity = float(speed if error > 0 else -speed)

    @property
    def moving(self) -> bool:
        if not self.torque_enabled:
            return False
        if self.mode == MODE_WHEEL:
            return self.wheel_speed != 0
        return abs(self.goal_position - self.position) >= 0.5

    def reading(self) -> ServoReading:
        return ServoReading(
            id=self.id,
            position=int(round(self.position)) % TICKS_PER_REV,
            goal_position=self.goal_position,
            speed=int(round(self.velocity)),
            goal_speed=self.wheel_speed if self.mode == MODE_WHEEL else self.goal_speed,
            acceleration=self.acc,
            load=self.load,
            voltage=self.voltage,
            temperature=self.temperature,
            current_ma=self.current_ma,
            moving=self.moving,
            torque_enabled=self.torque_enabled,
            status=self.status,
            errors=decode_errors(self.status),
        )


class FakeBus:
    """BusProtocol test double. All methods hold self.lock."""

    def __init__(self, port: str, baudrate: int = 1_000_000,
                 ids: Iterable[int] = (1, 2, 3, 4, 5, 6),
                 limits: dict[int, tuple[int, int]] | tuple[int, int] | None = None):
        self.port = port
        self.baudrate = baudrate
        self.lock = threading.RLock()
        self.port_present = True
        self.fail = False
        self._open = False
        self.servos: dict[int, FakeServo] = {}
        for sid in ids:
            s = FakeServo(id=sid)
            lim = limits.get(sid) if isinstance(limits, dict) else limits
            if lim is not None:
                s.min_limit, s.max_limit = lim
            self.servos[sid] = s
        # Call log for tests: list of (method name, args).
        self.calls: list[tuple[str, tuple]] = []

    # --- helpers -------------------------------------------------------------

    def _log(self, name: str, *args) -> None:
        self.calls.append((name, args))

    def calls_of(self, name: str) -> list[tuple]:
        with self.lock:
            return [args for n, args in self.calls if n == name]

    def _check(self) -> None:
        if self.fail:
            raise BusError(f"{self.port}: simulated link failure")
        if not self._open:
            raise BusError(f"{self.port}: port is not open")

    def _servo(self, sid: int) -> FakeServo:
        s = self.servos.get(sid)
        if s is None or not s.online:
            raise ServoError(f"servo {sid} did not answer")
        s.update(time.monotonic())
        return s

    def _tick_all(self) -> None:
        now = time.monotonic()
        for s in self.servos.values():
            s.update(now)

    # --- BusProtocol ---------------------------------------------------------

    def open(self) -> None:
        with self.lock:
            self._log("open")
            if not self.port_present or self.fail:
                raise BusError(f"{self.port}: no such port (simulated)")
            self._open = True

    def close(self) -> None:
        with self.lock:
            self._log("close")
            self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def ping(self, sid: int) -> int:
        with self.lock:
            self._check()
            self._servo(sid)
            return MODEL_STS3215

    def scan(self, ids: Iterable[int]) -> list[int]:
        with self.lock:
            self._check()
            return [i for i in ids if i in self.servos and self.servos[i].online]

    def read_states(self, ids: list[int]) -> dict[int, ServoReading]:
        with self.lock:
            self._check()
            self._log("read_states", tuple(ids))
            self._tick_all()
            out = {i: self.servos[i].reading() for i in ids
                   if i in self.servos and self.servos[i].online}
            if ids and not out:
                raise ServoError(f"no servo answered on {self.port}")
            return out

    def read_limits(self, sid: int) -> tuple[int, int]:
        with self.lock:
            self._check()
            s = self._servo(sid)
            return s.min_limit, s.max_limit

    def get_mode(self, sid: int) -> int:
        with self.lock:
            self._check()
            return self._servo(sid).mode

    def set_mode(self, sid: int, mode: int) -> None:
        with self.lock:
            self._check()
            self._log("set_mode", sid, mode)
            s = self._servo(sid)
            s.mode = mode
            if mode == MODE_WHEEL:
                s.wheel_speed = 0
            else:
                s.goal_position = int(round(s.position)) % TICKS_PER_REV

    def set_torque(self, ids: list[int], enabled: bool) -> None:
        with self.lock:
            self._check()
            self._log("set_torque", tuple(ids), enabled)
            self._tick_all()
            for sid in ids:
                s = self.servos.get(sid)
                if s is None or not s.online:
                    continue  # Sync write: no answer, no error.
                s.torque_enabled = enabled

    def write_positions(self, targets: dict[int, tuple[int, int, int]]) -> None:
        with self.lock:
            self._check()
            self._log("write_positions", dict(targets))
            self._tick_all()
            for sid, (pos, speed, acc) in targets.items():
                s = self.servos.get(sid)
                if s is None or not s.online:
                    continue
                s.goal_position = max(POSITION_MIN, min(POSITION_MAX, int(pos)))
                s.goal_speed = int(speed)
                s.acc = int(acc)

    def spin(self, sid: int, speed: int, acc: int) -> None:
        with self.lock:
            self._check()
            self._log("spin", sid, speed, acc)
            s = self._servo(sid)
            s.wheel_speed = max(-SPEED_MAX, min(SPEED_MAX, int(speed)))
            s.acc = int(acc)

    def hold(self, ids: list[int]) -> dict[int, int]:
        with self.lock:
            self._check()
            self._log("hold", tuple(ids))
            self._tick_all()
            out = {}
            for sid in ids:
                s = self.servos.get(sid)
                if s is None or not s.online:
                    continue
                pos = int(round(s.position)) % TICKS_PER_REV
                s.position = float(pos)
                s.goal_position = pos
                if s.mode == MODE_WHEEL:
                    s.wheel_speed = 0
                out[sid] = pos
            return out

    # Register map: some addresses map to the live state. Others use a dict.
    def read_register(self, sid: int, addr: int, size: int) -> int:
        with self.lock:
            self._check()
            s = self._servo(sid)
            live = {
                5: sid, 9: s.min_limit, 11: s.max_limit, 33: s.mode,
                40: int(s.torque_enabled), 41: s.acc, 42: s.goal_position,
                46: s.goal_speed, 55: s.lock_flag, 56: int(round(s.position)) % TICKS_PER_REV,
                62: int(s.voltage * 10), 63: s.temperature, 65: s.status, 66: int(s.moving),
            }
            value = live.get(addr, s.registers.get(addr, 0))
            return value & (0xFF if size == 1 else 0xFFFF)

    def write_register(self, sid: int, addr: int, size: int, value: int) -> None:
        with self.lock:
            self._check()
            self._log("write_register", sid, addr, size, value)
            s = self._servo(sid)
            if addr == 33:
                s.mode = value
            elif addr == 40:
                s.torque_enabled = bool(value)
            elif addr == 41:
                s.acc = value
            elif addr == 42:
                s.goal_position = value
            elif addr == 46:
                s.goal_speed = value
            elif addr == 55:
                s.lock_flag = value
            else:
                s.registers[addr] = value
