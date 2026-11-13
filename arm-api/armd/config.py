"""Load config.yaml into dataclasses.

Path order: the --config argument, else env ARMD_CONFIG, else config.yaml
next to the armd package. Env ARMD_ADMIN=1 turns on admin. Env ARMD_FAKE=1
turns on fake mode (FakeBus for all arms, no hardware).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import ACC_MAX, ERROR_FLAGS, POSITION_MAX, POSITION_MIN, SPEED_MAX

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_BLOCK_ON_ERRORS = ["overload", "temperature", "current", "sensor"]
ON_SHUTDOWN_VALUES = ("hold", "limp")
TRUE_VALUES = ("1", "true", "yes", "on")


class ConfigError(ValueError):
    """The config file has a bad value."""


@dataclass
class JointConfig:
    name: str
    id: int
    min: int | None = None            # ticks. None = use the EEPROM limit.
    max: int | None = None
    default_speed: int | None = None  # None = use the arm default.
    default_acc: int | None = None


@dataclass
class ArmConfig:
    name: str
    port: str
    baudrate: int = 1_000_000
    allow_motion: bool = False
    allow_spin: bool = False
    max_speed: int = 1000
    default_speed: int = 300
    default_acc: int = 30
    on_shutdown: str = "hold"
    block_on_errors: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCK_ON_ERRORS))
    joints: dict[str, JointConfig] = field(default_factory=dict)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    poll_hz: float = 30.0
    admin: bool = False
    fake: bool = False


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    arms: dict[str, ArmConfig] = field(default_factory=dict)
    path: str | None = None


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def resolve_config_path(cli_path: str | os.PathLike | None = None) -> Path:
    if cli_path:
        return Path(cli_path)
    env = os.environ.get("ARMD_CONFIG")
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_config(path: str | os.PathLike | None = None, *, fake: bool | None = None,
                admin: bool | None = None) -> Config:
    """Read the YAML file and return a validated Config.

    fake / admin: True from the command line turns the flag on. The env
    variables ARMD_FAKE / ARMD_ADMIN can also turn them on.
    """
    p = resolve_config_path(path)
    try:
        text = p.read_text()
    except OSError as e:
        raise ConfigError(f"cannot read config file {p}: {e}") from e
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"bad YAML in {p}: {e}") from e
    cfg = config_from_dict(data)
    cfg.path = str(p)
    if fake or env_flag("ARMD_FAKE"):
        cfg.server.fake = True
    if admin or env_flag("ARMD_ADMIN"):
        cfg.server.admin = True
    return cfg


# --- validation --------------------------------------------------------------

def _check_keys(where: str, data: dict, allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")


def _int(where: str, value: Any, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: must be an integer, not {value!r}")
    if lo is not None and value < lo:
        raise ConfigError(f"{where}: must be >= {lo}, not {value}")
    if hi is not None and value > hi:
        raise ConfigError(f"{where}: must be <= {hi}, not {value}")
    return value


def _bool(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: must be true or false, not {value!r}")
    return value


def _str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: must be a non-empty string, not {value!r}")
    return value


def _joint_from_dict(arm: str, name: str, data: Any) -> JointConfig:
    where = f"arms.{arm}.joints.{name}"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: must be a mapping like {{id: 1}}")
    _check_keys(where, data, {"id", "min", "max", "default_speed", "default_acc"})
    if "id" not in data:
        raise ConfigError(f"{where}: id is missing")
    j = JointConfig(name=name, id=_int(f"{where}.id", data["id"], 0, 253))
    if data.get("min") is not None:
        j.min = _int(f"{where}.min", data["min"], POSITION_MIN, POSITION_MAX)
    if data.get("max") is not None:
        j.max = _int(f"{where}.max", data["max"], POSITION_MIN, POSITION_MAX)
    if j.min is not None and j.max is not None and j.min >= j.max:
        raise ConfigError(f"{where}: min must be less than max")
    if data.get("default_speed") is not None:
        j.default_speed = _int(f"{where}.default_speed", data["default_speed"], 1, SPEED_MAX)
    if data.get("default_acc") is not None:
        j.default_acc = _int(f"{where}.default_acc", data["default_acc"], 0, ACC_MAX)
    return j


def _arm_from_dict(name: str, data: Any) -> ArmConfig:
    where = f"arms.{name}"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: must be a mapping")
    _check_keys(where, data, {
        "port", "baudrate", "allow_motion", "allow_spin", "max_speed", "default_speed",
        "default_acc", "on_shutdown", "block_on_errors", "joints"})
    if "port" not in data:
        raise ConfigError(f"{where}: port is missing")
    a = ArmConfig(name=name, port=_str(f"{where}.port", data["port"]))
    if "baudrate" in data:
        a.baudrate = _int(f"{where}.baudrate", data["baudrate"], 1)
    if "allow_motion" in data:
        a.allow_motion = _bool(f"{where}.allow_motion", data["allow_motion"])
    if "allow_spin" in data:
        a.allow_spin = _bool(f"{where}.allow_spin", data["allow_spin"])
    if "max_speed" in data:
        a.max_speed = _int(f"{where}.max_speed", data["max_speed"], 1, SPEED_MAX)
    if "default_speed" in data:
        a.default_speed = _int(f"{where}.default_speed", data["default_speed"], 1, SPEED_MAX)
    if "default_acc" in data:
        a.default_acc = _int(f"{where}.default_acc", data["default_acc"], 0, ACC_MAX)
    if "on_shutdown" in data:
        if data["on_shutdown"] not in ON_SHUTDOWN_VALUES:
            raise ConfigError(f"{where}.on_shutdown: must be one of {ON_SHUTDOWN_VALUES}")
        a.on_shutdown = data["on_shutdown"]
    if "block_on_errors" in data:
        errs = data["block_on_errors"] or []
        known = set(ERROR_FLAGS.values())
        if not isinstance(errs, list) or any(e not in known for e in errs):
            raise ConfigError(f"{where}.block_on_errors: must be a list of {sorted(known)}")
        a.block_on_errors = list(errs)
    joints = data.get("joints") or {}
    if not isinstance(joints, dict) or not joints:
        raise ConfigError(f"{where}.joints: must be a non-empty mapping")
    seen: dict[int, str] = {}
    for jname, jdata in joints.items():
        j = _joint_from_dict(name, str(jname), jdata)
        if j.id in seen:
            raise ConfigError(f"{where}.joints: id {j.id} is used by {seen[j.id]} and {j.name}")
        seen[j.id] = j.name
        a.joints[j.name] = j
    return a


def config_from_dict(data: Any) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("config: top level must be a mapping")
    _check_keys("config", data, {"server", "arms"})
    cfg = Config()
    server = data.get("server") or {}
    if not isinstance(server, dict):
        raise ConfigError("server: must be a mapping")
    _check_keys("server", server, {"host", "port", "poll_hz", "admin", "fake"})
    s = cfg.server
    if "host" in server:
        s.host = _str("server.host", server["host"])
    if "port" in server:
        s.port = _int("server.port", server["port"], 1, 65535)
    if "poll_hz" in server:
        hz = server["poll_hz"]
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not 0 < hz <= 200:
            raise ConfigError("server.poll_hz: must be a number in 0 < poll_hz <= 200")
        s.poll_hz = float(hz)
    if "admin" in server:
        s.admin = _bool("server.admin", server["admin"])
    if "fake" in server:
        s.fake = _bool("server.fake", server["fake"])
    arms = data.get("arms") or {}
    if not isinstance(arms, dict):
        raise ConfigError("arms: must be a mapping")
    for aname, adata in arms.items():
        cfg.arms[str(aname)] = _arm_from_dict(str(aname), adata)
    return cfg
