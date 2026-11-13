"""Bus: one serial port with many STS3215 servos on it (real hardware).

The Bus owns one PortHandler and one sms_sts packet handler (scservo_sdk).
It puts them in servo.Servo._connections[port], so every servo.Servo object
that it makes uses the same connection. Bus.close() removes that entry.

Which code does the work:
- servo.Servo (vendored library): ping, torque on/off, mode, angle limits,
  1- and 2-byte register read/write.
- scservo_sdk directly: the status block read (GroupSyncRead and readTxRx of
  addresses 40-70), the move sync write (GroupSyncWrite of addresses 41-47),
  the hold sync write (addresses 42-43), and WriteSpec for spin. Servo has no
  block read or sync write, and Servo.spin() reads the mode first (one more
  round trip).

Errors:
- BusError: the port is not open, the port is gone, or the link fails. After
  a serial exception the Bus closes itself, so the Arm poller can open it
  again.
- ServoError: one servo did not answer, or sent a bad packet.
- The error byte in a status packet is the same set of bits as the status
  register (address 65). It is not an error of the command, so the Bus only
  logs it. read_states() gives it in ServoReading.errors.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import operator
import os
import threading
from typing import Iterable, Iterator

import serial
from scservo_sdk import COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PortHandler, sms_sts

from . import registers as R
from .servo import Servo
from .types import (
    ACC_MAX,
    MODE_POSITION,
    MODE_STEP,
    MODE_WHEEL,
    POSITION_MAX,
    POSITION_MIN,
    SPEED_MAX,
    BusError,
    InvalidRequest,
    ServoError,
    ServoReading,
)

try:  # pyserial calls termios. termios.error is not an OSError.
    import termios

    _TERMIOS_ERRORS: tuple[type[BaseException], ...] = (termios.error,)
except ImportError:  # pragma: no cover - not Linux
    _TERMIOS_ERRORS = ()

log = logging.getLogger("armd.bus")

MAX_SERVO_ID = 252          # scservo_sdk MAX_ID. 253 and 254 are not servo ids.
SYNC_READ_STRIKES = 3       # After this many sync reads with no data, but single
                            # reads that work, use single reads only.


class _CommFailure(Exception):
    """A servo did not answer, or the packet was bad. Internal only."""

    def __init__(self, sid: int | None, what: str, result: int | None, text: str = ""):
        super().__init__(text or what)
        self.sid = sid
        self.what = what
        self.result = result
        self.text = text


class _BusServo(Servo):
    """servo.Servo with a different _check().

    Servo._check() raises IOError for a comm failure and also for a status
    error bit in the reply. A servo with (for example) a voltage error still
    does the command, so here only a comm failure is an error.
    """

    def _check(self, comm_result, error, what):
        if comm_result != COMM_SUCCESS:
            raise _CommFailure(self.servo_id, what, comm_result)
        if error:
            log.debug("servo %s: status bits 0x%02x in reply (%s)", self.servo_id, error, what)


def _int(name: str, value, lo: int, hi: int, sid: int | None = None) -> int:
    """Return value as int. Raise InvalidRequest if it is not an int in lo..hi."""
    where = f"servo {sid}: " if sid is not None else ""
    if isinstance(value, bool):
        raise InvalidRequest(f"{where}{name} must be an integer, got {value!r}")
    try:
        v = operator.index(value)
    except TypeError:
        raise InvalidRequest(f"{where}{name} must be an integer, got {value!r}") from None
    if not lo <= v <= hi:
        raise InvalidRequest(f"{where}{name} must be {lo}..{hi}, got {v}")
    return v


def _open_error_text(port: str, exc: BaseException) -> str:
    code = getattr(exc, "errno", None)
    if code in (errno.EACCES, errno.EPERM):
        return (f"permission denied on {port}. Add the service user to the 'dialout' group "
                f"(sudo usermod -aG dialout <user>), then log in again: {exc}")
    if code == errno.ENOENT:
        return f"serial port {port} does not exist (is the USB cable connected?): {exc}"
    if code == errno.EBUSY:
        return f"serial port {port} is busy (does another program use it?): {exc}"
    return f"could not open serial port {port}: {exc}"


class Bus:
    """Real STS3215 bus on one serial port. Implements types.BusProtocol."""

    def __init__(self, port: str, baudrate: int = 1_000_000):
        self.port = port
        self.baudrate = baudrate
        self.lock = threading.RLock()
        self._ph: PortHandler | None = None    # scservo_sdk PortHandler
        self._pk: sms_sts | None = None        # scservo_sdk packet handler
        self._servos: dict[int, _BusServo] = {}
        self._sync_read_enabled = True
        self._sync_read_strikes = 0

    def __repr__(self) -> str:
        return f"Bus({self.port!r}, baudrate={self.baudrate}, open={self.is_open})"

    # --- Open and close ------------------------------------------------------

    def open(self) -> None:
        with self.lock:
            if self._is_open():
                return
            self._close()  # Remove a half-closed state, if there is one.
            if not os.path.exists(self.port):
                raise BusError(f"serial port {self.port} does not exist (is the USB cable connected?)")
            if not os.access(self.port, os.R_OK | os.W_OK):
                raise BusError(f"permission denied on {self.port}. Add the service user to the "
                               f"'dialout' group (sudo usermod -aG dialout <user>), then log in again")
            if self.port in Servo._connections:
                raise BusError(f"serial port {self.port} is already open in this process")

            ph = PortHandler(self.port)
            try:
                ok = ph.openPort()
                if ok and ph.getBaudRate() != self.baudrate:
                    ok = ph.setBaudRate(self.baudrate)
            except (serial.SerialException, OSError, *_TERMIOS_ERRORS) as e:
                self._close_handler(ph)
                raise BusError(_open_error_text(self.port, e)) from e
            if not ok:
                self._close_handler(ph)
                raise BusError(f"serial port {self.port}: baud rate {self.baudrate} is not supported")

            self._ph = ph
            self._pk = sms_sts(ph)
            Servo._connections[self.port] = {"handler": self._ph, "servo": self._pk}
            self._sync_read_enabled = True
            self._sync_read_strikes = 0
            log.info("opened %s at %d baud", self.port, self.baudrate)

    def close(self) -> None:
        with self.lock:
            if self._ph is not None:
                log.info("closing %s", self.port)
            self._close()

    @property
    def is_open(self) -> bool:
        with self.lock:
            return self._is_open()

    # --- Simple per-servo operations (servo.Servo) ----------------------------

    def ping(self, sid: int) -> int:
        with self.lock:
            s = self._servo(sid)
            return int(self._one(sid, "ping", s.ping))

    def scan(self, ids: Iterable[int]) -> list[int]:
        with self.lock:
            found = []
            for sid in self._check_ids(ids):
                try:
                    self.ping(sid)
                except ServoError:
                    continue
                found.append(sid)
            return found

    def read_limits(self, sid: int) -> tuple[int, int]:
        with self.lock:
            s = self._servo(sid)
            lo = self._one(sid, "read min angle limit", s.read2Byte, R.ADDR_MIN_ANGLE_LIMIT)
            hi = self._one(sid, "read max angle limit", s.read2Byte, R.ADDR_MAX_ANGLE_LIMIT)
            return int(lo), int(hi)

    def get_mode(self, sid: int) -> int:
        with self.lock:
            s = self._servo(sid)
            return int(self._one(sid, "read mode", s.get_mode))

    def set_mode(self, sid: int, mode: int) -> None:
        with self.lock:
            mode = _int("mode", mode, MODE_POSITION, MODE_STEP, sid)
            s = self._servo(sid)
            self._one(sid, f"set mode {mode}", self._set_mode, s, mode)

    def _set_mode(self, s: _BusServo, mode: int) -> None:
        if s.get_mode() == mode:
            return
        was_on = s.torque_enabled()
        if was_on:
            s.disable_torque()
        s.set_mode(mode)
        # Make the new mode safe before torque comes on again.
        if mode == MODE_WHEEL:
            # The old goal speed would make the servo spin. Set it to 0 (stop).
            s.write2Byte(R.ADDR_GOAL_SPEED, 0)
        elif mode == MODE_POSITION:
            # After spin, the goal position is 0 (WriteSpec writes it).
            # Set goal = present so the servo does not move.
            present = R.sign_magnitude(s.read2Byte(R.ADDR_PRESENT_POSITION), R.POSITION_SIGN_BIT)
            s.write2Byte(R.ADDR_GOAL_POSITION, max(POSITION_MIN, min(POSITION_MAX, present)))
        # If a step above fails, torque stays off. That is the safe state.
        if was_on:
            s.enable_torque()

    def set_torque(self, ids: list[int], enabled: bool) -> None:
        with self.lock:
            ids = self._check_ids(ids)
            self._require_open()
            what = "torque on" if enabled else "torque off"
            failed: dict[int, _CommFailure] = {}
            # One write per servo (not a sync write), so that each servo
            # confirms. A failed servo does not stop the others.
            for sid in ids:
                s = self._servo(sid)
                try:
                    with self._io(f"{what} (servo {sid})"):
                        if enabled:
                            s.enable_torque()
                        else:
                            s.disable_torque()
                except _CommFailure as f:
                    failed[sid] = f
            self._raise_failures(what, ids, failed)

    def read_register(self, sid: int, addr: int, size: int) -> int:
        with self.lock:
            size = _int("size", size, 1, 2, sid)
            addr = _int("addr", addr, 0, 256 - size, sid)
            s = self._servo(sid)
            fn = s.read1Byte if size == 1 else s.read2Byte
            return int(self._one(sid, f"read addr {addr}", fn, addr))

    def write_register(self, sid: int, addr: int, size: int, value: int) -> None:
        with self.lock:
            size = _int("size", size, 1, 2, sid)
            addr = _int("addr", addr, 0, 256 - size, sid)
            value = _int("value", value, 0, 0xFF if size == 1 else 0xFFFF, sid)
            s = self._servo(sid)
            fn = s.write1Byte if size == 1 else s.write2Byte
            self._one(sid, f"write addr {addr}", fn, addr, value)

    # --- Multi-servo operations (scservo_sdk) ---------------------------------

    def read_states(self, ids: list[int]) -> dict[int, ServoReading]:
        with self.lock:
            ids = self._check_ids(ids)
            self._require_open()
            if not ids:
                return {}
            blocks: dict[int, bytes] = {}
            if self._sync_read_enabled:
                blocks = self._sync_read_blocks(ids)
            missing = [sid for sid in ids if sid not in blocks]
            if missing:
                single = {}
                for sid in missing:
                    data = self._read_block_one(sid)
                    if data is not None:
                        single[sid] = data
                if self._sync_read_enabled:
                    self._note_sync_read(got_sync=bool(blocks), got_single=bool(single))
                blocks.update(single)
                still = [sid for sid in missing if sid not in single]
                if still:
                    log.debug("%s: no answer from servo(s) %s", self.port, still)
            elif self._sync_read_enabled:
                self._sync_read_strikes = 0
            if not blocks:
                self._check_port_present("read states")
                raise ServoError(f"{self.port}: no servo answered (ids {ids})")
            return {sid: R.decode_block(sid, blocks[sid]) for sid in ids if sid in blocks}

    def write_positions(self, targets: dict[int, tuple[int, int, int]]) -> None:
        with self.lock:
            rows = []
            for sid, target in targets.items():
                sid = self._check_id(sid)
                try:
                    pos, speed, acc = target
                except (TypeError, ValueError):
                    raise InvalidRequest(
                        f"servo {sid}: target must be (position, speed, acc), got {target!r}") from None
                pos = _int("position", pos, POSITION_MIN, POSITION_MAX, sid)
                speed = _int("speed", speed, 0, SPEED_MAX, sid)
                acc = _int("acc", acc, 0, ACC_MAX, sid)
                # Addresses 41-47: acc, goal position (2), goal time (2) = 0, goal speed (2).
                rows.append((sid, [acc, pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8]))
            self._require_open()
            if not rows:
                return
            gsw = GroupSyncWrite(self._pk, R.MOVE_START, R.MOVE_LEN)
            for sid, data in rows:
                gsw.addParam(sid, data)
            self._sync_write(gsw, "write positions")

    def spin(self, sid: int, speed: int, acc: int) -> None:
        with self.lock:
            speed = _int("speed", speed, -SPEED_MAX, SPEED_MAX, sid)
            acc = _int("acc", acc, 0, ACC_MAX, sid)
            s = self._servo(sid)
            pk = self._pk

            def run():
                # WriteSpec writes addresses 41-47: acc, goal position 0,
                # goal time 0, and the speed in sign-magnitude (bit 15).
                result, error = pk.WriteSpec(sid, speed, acc)
                s._check(result, error, "WriteSpec")

            self._one(sid, "spin", run)

    def hold(self, ids: list[int]) -> dict[int, int]:
        with self.lock:
            ids = self._check_ids(ids)
            states = self.read_states(ids)
            positions = {sid: max(POSITION_MIN, min(POSITION_MAX, r.position))
                         for sid, r in states.items()}
            lost = [sid for sid in ids if sid not in states]
            if lost:
                log.warning("%s: hold: no answer from servo(s) %s", self.port, lost)

            # Write only the goal position (addresses 42-43). Goal speed and
            # acc do not change.
            gsw = GroupSyncWrite(self._pk, R.ADDR_GOAL_POSITION, 2)
            for sid, pos in positions.items():
                gsw.addParam(sid, [pos & 0xFF, pos >> 8])
            self._sync_write(gsw, "hold")

            # In wheel mode the goal position has no effect. Set the goal speed to 0.
            for sid in positions:
                s = self._servo(sid)
                try:
                    with self._io(f"hold (servo {sid})"):
                        if s.get_mode() == MODE_WHEEL:
                            s.write2Byte(R.ADDR_GOAL_SPEED, 0)
                except _CommFailure as f:
                    log.warning("%s: hold: servo %s: mode check failed: %s",
                                self.port, sid, self._result_text(f))
            return positions

    # --- Internal --------------------------------------------------------------

    def _is_open(self) -> bool:
        ph = self._ph
        if ph is None or not ph.is_open or ph.ser is None:
            return False
        return bool(getattr(ph.ser, "is_open", True))

    def _require_open(self) -> None:
        if not self._is_open():
            raise BusError(f"serial port {self.port} is not open")

    @staticmethod
    def _close_handler(ph: PortHandler) -> None:
        try:
            if ph.ser is not None:
                ph.closePort()
        except Exception as e:  # The port can be gone already.
            log.debug("close of %s failed: %s", ph.port_name, e)
        ph.is_open = False
        ph.is_using = False

    def _close(self) -> None:
        ph = self._ph
        entry = Servo._connections.get(self.port)
        if entry is not None and ph is not None and entry.get("handler") is ph:
            del Servo._connections[self.port]
        self._ph = None
        self._pk = None
        self._servos.clear()
        if ph is not None:
            self._close_handler(ph)

    def _link_lost(self, what: str, cause) -> None:
        """Close the bus and raise BusError. The poller opens it again later."""
        log.warning("%s: link failed during %s: %s. Closing the port.", self.port, what, cause)
        self._close()
        raise BusError(f"serial port {self.port}: link failed during {what}: {cause}")

    def _check_port_present(self, what: str) -> None:
        """Call this after a failure. If the device path is gone, the link is lost."""
        if not os.path.exists(self.port):
            self._link_lost(what, "the device is gone")

    @contextlib.contextmanager
    def _io(self, what: str) -> Iterator[None]:
        """Change serial exceptions to BusError (and close the bus)."""
        try:
            yield
        except serial.SerialException as e:
            self._link_lost(what, e)
        except _TERMIOS_ERRORS as e:
            self._link_lost(what, e)
        except OSError as e:
            if e.errno is None:
                # servo.Servo raises a plain IOError (no errno) when a servo
                # does not answer a ping.
                raise _CommFailure(None, what, None, str(e)) from None
            self._link_lost(what, e)
        finally:
            # After an exception, the SDK can leave is_using = True. Then all
            # later packets fail with COMM_PORT_BUSY. We hold the lock, so
            # nothing else uses the port now.
            if self._ph is not None:
                self._ph.is_using = False

    def _one(self, sid: int, what: str, fn, *args):
        """Do one per-servo operation. A comm failure gives ServoError."""
        try:
            with self._io(f"{what} (servo {sid})"):
                return fn(*args)
        except _CommFailure as f:
            self._check_port_present(what)
            raise ServoError(f"servo {sid}: {what} failed: {self._result_text(f)}") from None

    def _raise_failures(self, what: str, ids: list[int], failed: dict[int, _CommFailure]) -> None:
        if not failed:
            return
        self._check_port_present(what)
        detail = "; ".join(f"servo {sid}: {self._result_text(f)}" for sid, f in failed.items())
        if len(ids) > 1 and len(failed) == len(ids):
            raise BusError(f"serial port {self.port}: {what}: no servo answered ({detail})")
        raise ServoError(f"{what} failed for servo(s) {sorted(failed)}: {detail}")

    def _result_text(self, f: _CommFailure) -> str:
        if f.text:
            return f.text
        if f.result is None:
            return "no answer"
        text = ""
        if self._pk is not None:
            text = self._pk.getTxRxResult(f.result).replace("[TxRxResult] ", "")
        return f"{text or 'comm error'} (comm_result {f.result})"

    def _check_id(self, sid) -> int:
        return _int("servo id", sid, 0, MAX_SERVO_ID)

    def _check_ids(self, ids: Iterable[int]) -> list[int]:
        out: list[int] = []
        for sid in ids:
            sid = self._check_id(sid)
            if sid not in out:
                out.append(sid)
        return out

    def _servo(self, sid: int) -> _BusServo:
        """Return the servo.Servo object for sid. It uses the Bus connection."""
        sid = self._check_id(sid)
        self._require_open()
        s = self._servos.get(sid)
        if s is None:
            # Servo() opens its own port if there is no entry. Make sure the
            # entry is ours, so that it does not.
            entry = Servo._connections.get(self.port)
            if entry is None or entry.get("handler") is not self._ph:
                Servo._connections[self.port] = {"handler": self._ph, "servo": self._pk}
            s = _BusServo(servo_id=sid, port=self.port)
            self._servos[sid] = s
        return s

    def _sync_read_blocks(self, ids: list[int]) -> dict[int, bytes]:
        """One sync read of the status block. Return the blocks that came back."""
        gsr = GroupSyncRead(self._pk, R.BLOCK_START, R.BLOCK_LEN)
        for sid in ids:
            gsr.addParam(sid)
        with self._io("sync read"):
            result = gsr.txRxPacket()
        out = {}
        for sid in ids:
            ok, _error = gsr.isAvailable(sid, R.BLOCK_START, R.BLOCK_LEN)
            if ok:
                # data_dict[sid] is [error byte, data bytes...].
                out[sid] = bytes(gsr.data_dict[sid][1:1 + R.BLOCK_LEN])
        if result != COMM_SUCCESS:
            log.debug("%s: sync read result %s, got %d of %d servos",
                      self.port, result, len(out), len(ids))
        return out

    def _read_block_one(self, sid: int) -> bytes | None:
        """Read the status block of one servo. Return None if it does not answer."""
        with self._io(f"read servo {sid}"):
            data, result, _error = self._pk.readTxRx(sid, R.BLOCK_START, R.BLOCK_LEN)
        if result != COMM_SUCCESS or len(data) < R.BLOCK_LEN:
            return None
        return bytes(data[:R.BLOCK_LEN])

    def _note_sync_read(self, got_sync: bool, got_single: bool) -> None:
        """Stop sync reads if they give nothing, but single reads work."""
        if got_sync or not got_single:
            self._sync_read_strikes = 0
            return
        self._sync_read_strikes += 1
        if self._sync_read_strikes >= SYNC_READ_STRIKES:
            self._sync_read_enabled = False
            log.warning("%s: sync read gives no data, but single reads work. "
                        "Using single reads until the port opens again.", self.port)

    def _sync_write(self, gsw: GroupSyncWrite, what: str) -> None:
        """Send a sync write. Servos do not reply to it."""
        with self._io(what):
            result = gsw.txPacket()
        if result != COMM_SUCCESS:
            self._check_port_present(what)
            text = self._pk.getTxRxResult(result) if self._pk is not None else ""
            raise BusError(f"serial port {self.port}: {what}: sync write failed: {text} ({result})")
