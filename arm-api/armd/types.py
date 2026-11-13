"""Shared types, constants, and errors.

bus.py (real hardware), fake.py (test double), arm.py, and server.py all use
this module. Do not change a signature here without a change to all users.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol, runtime_checkable

# --- STS3215 constants -------------------------------------------------------

TICKS_PER_REV = 4096
POSITION_MIN = 0
POSITION_MAX = 4095
SPEED_MAX = 3400          # ticks/s
ACC_MAX = 254
MODEL_STS3215 = 777

MODE_POSITION = 0
MODE_WHEEL = 1
MODE_PWM = 2
MODE_STEP = 3

# Bit number in the status register (address 65) -> error name.
ERROR_FLAGS = {0: "voltage", 1: "sensor", 2: "temperature", 3: "current", 5: "overload"}

# EEPROM addresses are below this value. The API does not write to them.
EEPROM_END = 40


def ticks_to_degrees(ticks: int) -> float:
    """0-4095 ticks -> 0-360 degrees (2048 ticks = 180 degrees)."""
    return ticks * 360.0 / TICKS_PER_REV


def degrees_to_ticks(degrees: float) -> int:
    """0-360 degrees -> 0-4095 ticks, rounded and clamped."""
    return max(POSITION_MIN, min(POSITION_MAX, round(degrees * TICKS_PER_REV / 360.0)))


def decode_errors(status: int) -> list[str]:
    return [name for bit, name in ERROR_FLAGS.items() if status & (1 << bit)]


# --- Readings ----------------------------------------------------------------

@dataclass
class ServoReading:
    """One read of one servo. Bus.read_states() returns these."""

    id: int
    position: int            # present position, ticks 0-4095
    goal_position: int       # ticks
    speed: int               # present speed, signed ticks/s (+ = CW)
    goal_speed: int          # ticks/s
    acceleration: int        # 0-254 (setting, not feedback)
    load: float              # signed percent, -100..100
    voltage: float           # volts
    temperature: int         # deg C
    current_ma: float        # mA
    moving: bool
    torque_enabled: bool
    status: int              # raw error bitmask (address 65)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- Errors (server.py maps http_status and code to the HTTP response) -------

class ArmApiError(Exception):
    http_status = 500
    code = "internal_error"


class InvalidRequest(ArmApiError):      # bad value, unknown joint, etc.
    http_status = 400
    code = "invalid_request"


class NotFound(ArmApiError):            # unknown arm or joint
    http_status = 404
    code = "not_found"


class MotionDisabled(ArmApiError):      # allow_motion / allow_spin / admin is off
    http_status = 403
    code = "motion_disabled"


class JointFault(ArmApiError):          # servo reports a blocking error
    http_status = 409
    code = "joint_fault"


class BusReleased(ArmApiError):         # the arm was released with /release
    http_status = 409
    code = "bus_released"


class ServoError(ArmApiError):          # servo did not answer, or sent an error packet
    http_status = 502
    code = "servo_error"


class BusError(ArmApiError):            # port missing, closed, or comm failure
    http_status = 503
    code = "bus_error"


# --- Bus interface -----------------------------------------------------------

@runtime_checkable
class BusProtocol(Protocol):
    """One serial port with many servos on it.

    All methods are thread-safe: each one holds self.lock (a threading.RLock)
    for its full duration. A caller can also hold self.lock to make a
    sequence of calls atomic.

    Errors: BusError if the port is not open or the link fails.
    ServoError if one servo does not answer or sends an error packet.
    """

    port: str
    lock: "object"  # threading.RLock

    def open(self) -> None: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...

    def ping(self, sid: int) -> int:
        """Return the model number (777 for STS3215)."""
    def scan(self, ids: Iterable[int]) -> list[int]:
        """Ping each id. Return the ids that answered."""
    def read_states(self, ids: list[int]) -> dict[int, ServoReading]:
        """Read addresses 40-70 of all ids with one sync read (fall back to
        one read per servo if sync read fails). A servo that does not answer
        is missing from the result (no exception), unless all are missing:
        then raise ServoError."""
    def read_limits(self, sid: int) -> tuple[int, int]:
        """Return (min, max) angle limits from EEPROM (addresses 9 and 11)."""
    def get_mode(self, sid: int) -> int: ...
    def set_mode(self, sid: int, mode: int) -> None:
        """Write address 33. Disable torque first if necessary, and restore it."""
    def set_torque(self, ids: list[int], enabled: bool) -> None: ...
    def write_positions(self, targets: dict[int, tuple[int, int, int]]) -> None:
        """{id: (position, speed, acc)}. One sync write of acc/position/time/speed
        (addresses 41-47). Does not change mode or torque."""
    def spin(self, sid: int, speed: int, acc: int) -> None:
        """Wheel mode speed, -3400..3400. Caller makes sure mode is MODE_WHEEL."""
    def hold(self, ids: list[int]) -> dict[int, int]:
        """Set goal position = present position (stop in place). Return the positions."""
    def read_register(self, sid: int, addr: int, size: int) -> int:
        """size is 1 or 2."""
    def write_register(self, sid: int, addr: int, size: int, value: int) -> None: ...
