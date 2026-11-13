"""STS3215 register map and decode of the status block.

Addresses are from the Feetech STS3215 memory table. They are the same as
the SMS_STS_* names in scservo_sdk/sms_sts.py and the ADDR_* names in
armd/servo.py. Two-byte values are little-endian (low byte first).
"""

from __future__ import annotations

from .types import ServoReading, decode_errors

# --- EEPROM (saved when the lock register is 0) ------------------------------

ADDR_MODEL = 3              # 2 bytes, read only. 777 = STS3215.
ADDR_ID = 5                 # 1 byte
ADDR_BAUD_RATE = 6          # 1 byte. 0 = 1 Mbaud.
ADDR_MIN_ANGLE_LIMIT = 9    # 2 bytes, ticks
ADDR_MAX_ANGLE_LIMIT = 11   # 2 bytes, ticks
ADDR_MODE = 33              # 1 byte. 0 position, 1 wheel, 2 PWM, 3 step.

# --- RAM (SRAM) ----------------------------------------------------------------

ADDR_TORQUE_ENABLE = 40     # 1 byte. 0 off, 1 on.
ADDR_ACCELERATION = 41      # 1 byte, 0-254
ADDR_GOAL_POSITION = 42     # 2 bytes, ticks
ADDR_GOAL_TIME = 44         # 2 bytes. armd always writes 0.
ADDR_GOAL_SPEED = 46        # 2 bytes, ticks/s. Bit 15 = sign (wheel mode).
ADDR_TORQUE_LIMIT = 48      # 2 bytes, 0-1000 (0.1 % units)
ADDR_LOCK = 55              # 1 byte. 1 = EEPROM writes are not saved.
ADDR_PRESENT_POSITION = 56  # 2 bytes, ticks. Bit 15 = sign.
ADDR_PRESENT_SPEED = 58     # 2 bytes, ticks/s. Bit 15 = sign.
ADDR_PRESENT_LOAD = 60      # 2 bytes. Bit 10 = direction, bits 0-9 = 0.1 %.
ADDR_PRESENT_VOLTAGE = 62   # 1 byte, 0.1 V units
ADDR_PRESENT_TEMPERATURE = 63  # 1 byte, deg C
ADDR_STATUS = 65            # 1 byte, error bits (see types.ERROR_FLAGS)
ADDR_MOVING = 66            # 1 byte. 1 = moving.
ADDR_PRESENT_CURRENT = 69   # 2 bytes, 6.5 mA units. Bit 15 = sign.

# The status block: one read of addresses 40..70 gives all of ServoReading.
BLOCK_START = ADDR_TORQUE_ENABLE
BLOCK_END = ADDR_PRESENT_CURRENT + 1      # last address in the block (70)
BLOCK_LEN = BLOCK_END - BLOCK_START + 1   # 31 bytes

# The sync write for moves: acc, goal position, goal time, goal speed.
MOVE_START = ADDR_ACCELERATION            # 41
MOVE_LEN = 7                              # addresses 41-47

# --- Scaling (the same as armd/servo.py) ------------------------------------

VOLTS_PER_UNIT = 0.1        # servo.py voltage(): raw * 0.1
MA_PER_UNIT = 6.5           # servo.py current_draw(): raw * 6.5
LOAD_PERCENT_PER_UNIT = 0.1  # servo.py current_load(): raw / 1000 * 100

SPEED_SIGN_BIT = 15
POSITION_SIGN_BIT = 15
CURRENT_SIGN_BIT = 15
LOAD_DIRECTION_BIT = 10


def sign_magnitude(raw: int, bit: int) -> int:
    """Decode a sign-magnitude value. If `bit` is set, the value is negative.

    This is the same as scs_tohost() in scservo_sdk.
    """
    mask = 1 << bit
    if raw & mask:
        return -(raw & (mask - 1))
    return raw & (mask - 1)


def to_sign_magnitude(value: int, bit: int) -> int:
    """Encode a signed value as sign-magnitude. Same as scs_toscs() in scservo_sdk."""
    if value < 0:
        return (-value) | (1 << bit)
    return value


def decode_load(raw: int) -> float:
    """Present load (address 60) -> signed percent, -100..100.

    Bits 0-9 are the magnitude in 0.1 % units. Bit 10 is the direction.
    armd/servo.py says bit 10 set = "CW", and that while the servo moves on
    its own the load direction is the direction of travel. Present speed is
    positive for CW. Thus bit 10 set gives a positive load here, so that load
    and speed have the same sign during a free move.
    """
    magnitude = (raw & 0x3FF) * LOAD_PERCENT_PER_UNIT
    value = magnitude if raw & (1 << LOAD_DIRECTION_BIT) else -magnitude
    return round(value, 1) + 0.0  # "+ 0.0" changes -0.0 to 0.0


def _word(data: bytes, addr: int) -> int:
    i = addr - BLOCK_START
    return data[i] | (data[i + 1] << 8)


def _byte(data: bytes, addr: int) -> int:
    return data[addr - BLOCK_START]


def decode_block(sid: int, data: bytes) -> ServoReading:
    """Decode the 31 bytes at addresses 40..70 of one servo.

    `data` can be bytes, bytearray, or a list of ints. Only the first 31
    items are used. Raise ValueError if there are fewer than 31.
    """
    if len(data) < BLOCK_LEN:
        raise ValueError(f"servo {sid}: block has {len(data)} bytes, need {BLOCK_LEN}")
    data = bytes(data[:BLOCK_LEN])

    status = _byte(data, ADDR_STATUS)
    current_raw = _word(data, ADDR_PRESENT_CURRENT)
    return ServoReading(
        id=sid,
        position=sign_magnitude(_word(data, ADDR_PRESENT_POSITION), POSITION_SIGN_BIT),
        goal_position=sign_magnitude(_word(data, ADDR_GOAL_POSITION), POSITION_SIGN_BIT),
        speed=sign_magnitude(_word(data, ADDR_PRESENT_SPEED), SPEED_SIGN_BIT),
        goal_speed=sign_magnitude(_word(data, ADDR_GOAL_SPEED), SPEED_SIGN_BIT),
        acceleration=_byte(data, ADDR_ACCELERATION),
        load=decode_load(_word(data, ADDR_PRESENT_LOAD)),
        voltage=round(_byte(data, ADDR_PRESENT_VOLTAGE) * VOLTS_PER_UNIT, 1),
        temperature=_byte(data, ADDR_PRESENT_TEMPERATURE),
        current_ma=round(sign_magnitude(current_raw, CURRENT_SIGN_BIT) * MA_PER_UNIT, 1) + 0.0,
        moving=bool(_byte(data, ADDR_MOVING)),
        torque_enabled=bool(_byte(data, ADDR_TORQUE_ENABLE)),
        status=status,
        errors=decode_errors(status),
    )
