"""Tests for armd.registers and armd.bus. No hardware is necessary.

The fake is at the wire level: FakePortHandler is a real scservo_sdk
PortHandler, but it sends the bytes to FakeWire, not to a serial port.
FakeWire acts as the servos. Thus the real sms_sts, GroupSyncRead and
GroupSyncWrite code runs in these tests.
"""

from __future__ import annotations

import errno

import pytest
import serial
import scservo_sdk.port_handler as sdk_port_handler
from scservo_sdk import PortHandler

import armd.bus as bus_mod
from armd import registers as R
from armd.bus import Bus
from armd.servo import Servo
from armd.types import (
    MODE_POSITION,
    MODE_WHEEL,
    BusError,
    BusProtocol,
    InvalidRequest,
    ServoError,
)

INST_PING, INST_READ, INST_WRITE = 1, 2, 3
INST_SYNC_READ, INST_SYNC_WRITE = 0x82, 0x83
BROADCAST = 0xFE


# --- Fake wire ---------------------------------------------------------------

def _w(mem: bytearray, addr: int, value: int) -> None:
    mem[addr] = value & 0xFF
    mem[addr + 1] = (value >> 8) & 0xFF


def _r(mem: bytearray, addr: int) -> int:
    return mem[addr] | (mem[addr + 1] << 8)


def make_servo(position: int = 2048) -> bytearray:
    mem = bytearray(256)
    _w(mem, R.ADDR_MODEL, 777)
    _w(mem, R.ADDR_MIN_ANGLE_LIMIT, 100)
    _w(mem, R.ADDR_MAX_ANGLE_LIMIT, 4000)
    mem[R.ADDR_TORQUE_ENABLE] = 1
    mem[R.ADDR_ACCELERATION] = 20
    _w(mem, R.ADDR_GOAL_POSITION, position)
    _w(mem, R.ADDR_GOAL_SPEED, 700)
    _w(mem, R.ADDR_PRESENT_POSITION, position)
    mem[R.ADDR_PRESENT_VOLTAGE] = 120
    mem[R.ADDR_PRESENT_TEMPERATURE] = 30
    return mem


class FakeWire:
    """Servos on a bus. It answers packets like STS3215 servos do."""

    def __init__(self, servos: dict[int, bytearray]):
        self.servos = servos
        self.reply_error: dict[int, int] = {}  # error byte in the status packets
        self.sync_read_supported = True
        self.fail_with: BaseException | None = None  # raise this on the next write
        self.rx = bytearray()
        self.packets: list[tuple[int, int, bytes]] = []  # (id, instruction, params)

    def instructions(self) -> list[int]:
        return [inst for _sid, inst, _p in self.packets]

    def _reply(self, sid: int, data: bytes = b"") -> None:
        pkt = [0xFF, 0xFF, sid, len(data) + 2, self.reply_error.get(sid, 0), *data]
        pkt.append(~sum(pkt[2:]) & 0xFF)
        self.rx.extend(pkt)

    def handle(self, packet) -> None:
        if self.fail_with is not None:
            exc, self.fail_with = self.fail_with, None
            raise exc
        p = bytes(packet)
        assert p[0] == 0xFF and p[1] == 0xFF
        sid, length, inst = p[2], p[3], p[4]
        params = p[5:5 + length - 2]
        assert (~sum(p[2:4 + length - 1]) & 0xFF) == p[3 + length], "bad checksum"
        self.packets.append((sid, inst, params))
        mem = self.servos.get(sid)
        if inst == INST_PING and mem is not None:
            self._reply(sid)
        elif inst == INST_READ and mem is not None:
            addr, n = params[0], params[1]
            self._reply(sid, bytes(mem[addr:addr + n]))
        elif inst == INST_WRITE and mem is not None:
            addr = params[0]
            mem[addr:addr + len(params) - 1] = params[1:]
            if sid != BROADCAST:
                self._reply(sid)
        elif inst == INST_SYNC_READ and self.sync_read_supported:
            addr, n = params[0], params[1]
            for one in params[2:]:
                if one in self.servos:
                    self._reply(one, bytes(self.servos[one][addr:addr + n]))
        elif inst == INST_SYNC_WRITE:
            addr, n = params[0], params[1]
            rest = params[2:]
            for i in range(0, len(rest), n + 1):
                one, data = rest[i], rest[i + 1:i + 1 + n]
                if one in self.servos:
                    self.servos[one][addr:addr + n] = data


class FakeSerial:
    def __init__(self):
        self.is_open = True

    def close(self):
        self.is_open = False


class FakePortHandler(PortHandler):
    """Real PortHandler, but the bytes go to a FakeWire."""

    wire: FakeWire

    def setupPort(self, cflag_baud):
        self.ser = FakeSerial()
        self.is_open = True
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    def closePort(self):
        self.ser.close()
        self.is_open = False

    def clearPort(self):
        self.wire.rx.clear()

    def getBytesAvailable(self):
        return len(self.wire.rx)

    def readPort(self, length):
        data = bytes(self.wire.rx[:length])
        del self.wire.rx[:length]
        return data

    def writePort(self, packet):
        self.wire.handle(packet)
        return len(packet)


@pytest.fixture
def wire():
    return FakeWire({sid: make_servo(2000 + sid) for sid in range(1, 7)})


@pytest.fixture
def port_path(tmp_path):
    p = tmp_path / "ttyFAKE0"
    p.touch()
    return str(p)


@pytest.fixture
def bus(monkeypatch, wire, port_path):
    # A short packet timeout, so tests with a missing servo are fast.
    monkeypatch.setattr(sdk_port_handler, "LATENCY_TIMER", 3)

    def factory(name):
        ph = FakePortHandler(name)
        ph.wire = wire
        return ph

    monkeypatch.setattr(bus_mod, "PortHandler", factory)
    b = Bus(port_path)
    b.open()
    yield b
    b.close()
    assert port_path not in Servo._connections


IDS = [1, 2, 3, 4, 5, 6]


# --- registers.decode_block ------------------------------------------------------

def block(**values) -> bytearray:
    """A 31-byte block (addresses 40..70). Keys are register addresses as 'a56' etc."""
    mem = bytearray(256)
    for key, (value, size) in values.items():
        addr = int(key[1:])
        if size == 1:
            mem[addr] = value
        else:
            _w(mem, addr, value)
    return mem[R.BLOCK_START:R.BLOCK_START + R.BLOCK_LEN]


def test_block_constants():
    assert (R.BLOCK_START, R.BLOCK_END, R.BLOCK_LEN) == (40, 70, 31)
    assert (R.MOVE_START, R.MOVE_LEN) == (41, 7)


def test_decode_block_fields():
    data = block(
        a40=(1, 1), a41=(50, 1), a42=(2100, 2), a46=(500, 2),
        a56=(2048, 2), a58=(0x8000 | 100, 2), a60=(0x400 | 250, 2),
        a62=(121, 1), a63=(35, 1), a65=(0b100101, 1), a66=(1, 1), a69=(10, 2),
    )
    r = R.decode_block(3, bytes(data))
    assert r.id == 3
    assert r.torque_enabled is True
    assert r.acceleration == 50
    assert r.goal_position == 2100
    assert r.goal_speed == 500
    assert r.position == 2048
    assert r.speed == -100
    assert r.load == 25.0
    assert r.voltage == 12.1
    assert r.temperature == 35
    assert r.status == 0b100101
    assert r.errors == ["voltage", "temperature", "overload"]
    assert r.moving is True
    assert r.current_ma == 65.0


def test_decode_block_signs():
    data = block(a58=(300, 2), a60=(300, 2), a69=(0x8000 | 20, 2), a46=(0x8000 | 800, 2))
    r = R.decode_block(1, list(data))  # a list of ints is also accepted
    assert r.speed == 300
    assert r.load == -30.0          # bit 10 clear = CCW = negative
    assert r.current_ma == -130.0   # bit 15 set = negative, 6.5 mA per unit
    assert r.goal_speed == -800     # wheel mode goal speed
    assert r.errors == [] and r.status == 0
    assert r.torque_enabled is False and r.moving is False


def test_decode_block_zero_load_is_not_negative_zero():
    r = R.decode_block(1, bytes(31))
    assert str(r.load) == "0.0" and str(r.current_ma) == "0.0"


def test_decode_block_too_short():
    with pytest.raises(ValueError, match="servo 4"):
        R.decode_block(4, bytes(30))


def test_sign_magnitude_round_trip():
    for v in (-3400, -1, 0, 1, 3400):
        assert R.sign_magnitude(R.to_sign_magnitude(v, 15), 15) == v
    assert R.to_sign_magnitude(-300, 15) == 0x8000 | 300
    assert R.sign_magnitude(0x8000, 15) == 0


# --- Bus: open and close ----------------------------------------------------------

def test_bus_is_a_bus_protocol(port_path):
    assert isinstance(Bus(port_path), BusProtocol)


def test_closed_bus_raises_bus_error(port_path):
    b = Bus(port_path)
    assert not b.is_open
    for call in (lambda: b.read_states(IDS), lambda: b.ping(1),
                 lambda: b.write_positions({1: (2048, 100, 10)}),
                 lambda: b.set_torque([1], False), lambda: b.hold([1])):
        with pytest.raises(BusError, match="not open"):
            call()
    b.close()
    b.close()  # safe to call two times


def test_open_missing_device(tmp_path):
    with pytest.raises(BusError, match="does not exist"):
        Bus(str(tmp_path / "ttyACM9")).open()


def test_open_permission_denied(monkeypatch, port_path):
    monkeypatch.setattr(bus_mod.os, "access", lambda *a: False)
    with pytest.raises(BusError, match="dialout"):
        Bus(port_path).open()


def test_open_serial_exception_eacces(monkeypatch, port_path):
    class Denied(FakePortHandler):
        def openPort(self):
            raise serial.SerialException(errno.EACCES, "could not open port: Permission denied")

    monkeypatch.setattr(bus_mod, "PortHandler", Denied)
    b = Bus(port_path)
    with pytest.raises(BusError, match="dialout"):
        b.open()
    assert not b.is_open
    assert port_path not in Servo._connections


def test_servo_shares_bus_connection(bus, port_path):
    entry = Servo._connections[port_path]
    assert entry["handler"] is bus._ph and entry["servo"] is bus._pk
    s = bus._servo(1)
    assert s._port_handler is bus._ph and s._servo is bus._pk
    bus.close()
    assert port_path not in Servo._connections
    assert not bus.is_open
    bus.close()
    bus.open()
    assert bus.is_open and Servo._connections[port_path]["handler"] is bus._ph


# --- Bus: read_states ------------------------------------------------------------

def test_read_states_one_sync_read(bus, wire):
    states = bus.read_states(IDS)
    assert sorted(states) == IDS
    assert wire.instructions() == [INST_SYNC_READ]
    r = states[4]
    assert r.position == 2004 and r.goal_speed == 700 and r.acceleration == 20
    assert r.voltage == 12.0 and r.temperature == 30 and r.torque_enabled


def test_read_states_falls_back_to_single_reads(bus, wire):
    wire.sync_read_supported = False
    states = bus.read_states(IDS)
    assert sorted(states) == IDS
    assert states[2].position == 2002
    assert wire.instructions() == [INST_SYNC_READ] + [INST_READ] * 6

    # After three such polls, the Bus stops the sync read.
    bus.read_states(IDS)
    bus.read_states(IDS)
    wire.packets.clear()
    bus.read_states(IDS)
    assert wire.instructions() == [INST_READ] * 6


def test_read_states_missing_servo_is_omitted(bus, wire):
    del wire.servos[5]
    states = bus.read_states(IDS)
    assert sorted(states) == [1, 2, 3, 4, 6]
    # Sync read got 5 servos. Only servo 5 was read again.
    assert wire.instructions() == [INST_SYNC_READ, INST_READ]
    assert bus._sync_read_enabled


def test_read_states_all_missing_raises_servo_error(bus, wire):
    wire.servos.clear()
    with pytest.raises(ServoError):
        bus.read_states(IDS)
    assert bus.is_open  # the port itself is good


def test_read_states_device_gone_raises_bus_error(bus, wire, port_path):
    import os
    wire.servos.clear()
    os.unlink(port_path)
    with pytest.raises(BusError, match="gone"):
        bus.read_states(IDS)
    assert not bus.is_open
    assert port_path not in Servo._connections


def test_serial_exception_closes_bus(bus, wire, port_path):
    wire.fail_with = serial.SerialException("device reports readiness to read but returned no data")
    with pytest.raises(BusError, match="link failed"):
        bus.read_states(IDS)
    assert not bus.is_open
    assert port_path not in Servo._connections
    with pytest.raises(BusError, match="not open"):
        bus.read_states(IDS)
    bus.open()  # The poller can open it again.
    assert sorted(bus.read_states(IDS)) == IDS


def test_os_error_closes_bus(bus, wire):
    wire.fail_with = OSError(errno.EIO, "Input/output error")
    with pytest.raises(BusError):
        bus.set_torque([1], False)
    assert not bus.is_open


def test_status_bits_in_reply_are_not_errors(bus, wire):
    wire.reply_error[1] = 0x01  # voltage error bit
    wire.servos[1][R.ADDR_STATUS] = 0x01
    bus.set_torque([1], False)
    assert wire.servos[1][R.ADDR_TORQUE_ENABLE] == 0
    assert bus.read_states([1])[1].errors == ["voltage"]


# --- Bus: write_positions ----------------------------------------------------------

@pytest.mark.parametrize("target", [
    (4096, 100, 10), (-1, 100, 10), (2048, 3401, 10), (2048, -1, 10),
    (2048, 100, 255), (2048, 100, -1), (2048.5, 100, 10), (True, 100, 10), (2048, 100),
])
def test_write_positions_rejects_bad_values(bus, wire, target):
    with pytest.raises(InvalidRequest):
        bus.write_positions({1: (2048, 100, 10), 2: target})
    assert wire.packets == []  # nothing was sent


def test_write_positions_bad_id(bus):
    with pytest.raises(InvalidRequest):
        bus.write_positions({300: (2048, 100, 10)})


def test_write_positions_one_sync_write(bus, wire):
    bus.write_positions({1: (1000, 300, 10), 6: (4095, 3400, 254)})
    assert wire.instructions() == [INST_SYNC_WRITE]
    m1, m6 = wire.servos[1], wire.servos[6]
    assert (m1[R.ADDR_ACCELERATION], _r(m1, R.ADDR_GOAL_POSITION), _r(m1, R.ADDR_GOAL_SPEED)) == (10, 1000, 300)
    assert _r(m1, R.ADDR_GOAL_TIME) == 0
    assert (m6[R.ADDR_ACCELERATION], _r(m6, R.ADDR_GOAL_POSITION), _r(m6, R.ADDR_GOAL_SPEED)) == (254, 4095, 3400)
    assert _r(wire.servos[2], R.ADDR_GOAL_POSITION) == 2002  # not in targets
    assert m1[R.ADDR_TORQUE_ENABLE] == 1  # torque does not change


def test_write_positions_empty_is_noop(bus, wire):
    bus.write_positions({})
    assert wire.packets == []


# --- Bus: other operations -----------------------------------------------------------

def test_ping_and_scan(bus, wire):
    assert bus.ping(1) == 777
    del wire.servos[3]
    with pytest.raises(ServoError, match="servo 3"):
        bus.ping(3)
    assert bus.scan(range(1, 8)) == [1, 2, 4, 5, 6]


def test_read_limits_and_mode(bus, wire):
    assert bus.read_limits(2) == (100, 4000)
    assert bus.get_mode(2) == MODE_POSITION
    del wire.servos[2]
    with pytest.raises(ServoError, match="servo 2"):
        bus.read_limits(2)


def test_set_mode_to_wheel_and_back(bus, wire):
    m = wire.servos[1]
    bus.set_mode(1, MODE_WHEEL)
    assert m[R.ADDR_MODE] == MODE_WHEEL
    assert _r(m, R.ADDR_GOAL_SPEED) == 0          # no spin when torque comes back
    assert m[R.ADDR_TORQUE_ENABLE] == 1           # torque restored
    writes = [p for sid, inst, p in wire.packets if inst == INST_WRITE]
    assert writes[0][0] == R.ADDR_TORQUE_ENABLE and writes[0][1] == 0  # torque off first

    bus.spin(1, -300, 5)
    assert _r(m, R.ADDR_GOAL_SPEED) == 0x8000 | 300
    assert m[R.ADDR_ACCELERATION] == 5
    assert _r(m, R.ADDR_GOAL_POSITION) == 0       # WriteSpec writes 0 here

    _w(m, R.ADDR_PRESENT_POSITION, 1234)
    bus.set_mode(1, MODE_POSITION)
    assert m[R.ADDR_MODE] == MODE_POSITION
    assert _r(m, R.ADDR_GOAL_POSITION) == 1234    # goal = present, no jump
    assert m[R.ADDR_TORQUE_ENABLE] == 1


def test_set_mode_torque_off_stays_off(bus, wire):
    m = wire.servos[2]
    m[R.ADDR_TORQUE_ENABLE] = 0
    bus.set_mode(2, MODE_WHEEL)
    assert m[R.ADDR_MODE] == MODE_WHEEL and m[R.ADDR_TORQUE_ENABLE] == 0


def test_set_mode_bad_value(bus):
    with pytest.raises(InvalidRequest):
        bus.set_mode(1, 4)


def test_spin_range(bus):
    with pytest.raises(InvalidRequest):
        bus.spin(1, 3401, 0)
    with pytest.raises(InvalidRequest):
        bus.spin(1, 100, 255)


def test_set_torque(bus, wire):
    bus.set_torque([1, 2], False)
    assert wire.servos[1][R.ADDR_TORQUE_ENABLE] == 0 and wire.servos[2][R.ADDR_TORQUE_ENABLE] == 0
    assert wire.servos[3][R.ADDR_TORQUE_ENABLE] == 1

    del wire.servos[2]
    with pytest.raises(ServoError, match=r"\[2\]"):
        bus.set_torque([1, 2], True)
    assert wire.servos[1][R.ADDR_TORQUE_ENABLE] == 1  # servo 1 still got it

    wire.servos.clear()
    with pytest.raises(BusError, match="no servo answered"):
        bus.set_torque([1, 2], False)


def test_hold(bus, wire):
    m1, m2 = wire.servos[1], wire.servos[2]
    _w(m1, R.ADDR_GOAL_POSITION, 3000)   # moving to 3000
    _w(m1, R.ADDR_PRESENT_POSITION, 2500)
    m2[R.ADDR_MODE] = MODE_WHEEL
    _w(m2, R.ADDR_GOAL_SPEED, 400)       # spinning
    del wire.servos[3]

    positions = bus.hold([1, 2, 3])
    assert positions == {1: 2500, 2: 2002}
    assert _r(m1, R.ADDR_GOAL_POSITION) == 2500
    assert _r(m1, R.ADDR_GOAL_SPEED) == 700 and m1[R.ADDR_ACCELERATION] == 20  # kept
    assert _r(m2, R.ADDR_GOAL_SPEED) == 0  # wheel mode: stopped


def test_registers(bus, wire):
    bus.write_register(1, R.ADDR_GOAL_SPEED, 2, 1500)
    assert bus.read_register(1, R.ADDR_GOAL_SPEED, 2) == 1500
    bus.write_register(1, R.ADDR_ACCELERATION, 1, 77)
    assert bus.read_register(1, R.ADDR_ACCELERATION, 1) == 77
    for bad in [dict(size=3, value=1), dict(size=1, value=256), dict(size=2, value=-1)]:
        with pytest.raises(InvalidRequest):
            bus.write_register(1, 41, bad["size"], bad["value"])
    with pytest.raises(InvalidRequest):
        bus.read_register(1, 41, 4)
    del wire.servos[1]
    with pytest.raises(ServoError, match="servo 1"):
        bus.read_register(1, 41, 1)


def test_lock_is_reentrant(bus):
    with bus.lock:
        bus.read_states([1])
        bus.write_positions({1: (2048, 100, 0)})
