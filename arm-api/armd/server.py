"""FastAPI app for armd. See ../arm-api.md for the HTTP API.

Endpoints that use the bus are sync `def`, so FastAPI runs them in its
thread pool and a slow servo does not block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Callable, Literal

from fastapi import Body, FastAPI, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .arm import Arm
from .config import ArmConfig, Config
from .types import ArmApiError, BusProtocol, NotFound

log = logging.getLogger("armd.server")

BusFactory = Callable[[ArmConfig], BusProtocol]


# --- request bodies ----------------------------------------------------------

class _Body(BaseModel):
    # Unknown fields are an error. So a typo (for example "positon") does not
    # become a silent default.
    model_config = ConfigDict(extra="forbid")


class JointTarget(_Body):
    position: int | None = None
    degrees: float | None = None
    delta: int | None = None
    delta_degrees: float | None = None
    speed: int | None = None
    acc: int | None = None


class PositionBody(JointTarget):
    wait: bool = False
    timeout: float = 5.0
    tolerance: int = 10


class PositionsBody(_Body):
    joints: dict[str, JointTarget]
    speed: int | None = None
    acc: int | None = None
    wait: bool = False
    timeout: float = 5.0
    tolerance: int = 10


class SpeedBody(_Body):
    speed: int
    acc: int | None = None


class TorqueBody(_Body):
    enabled: bool
    joints: list[str] | None = None


class JointTorqueBody(_Body):
    enabled: bool


class StopBody(_Body):
    mode: Literal["hold", "limp"] = "hold"


class RegisterWriteBody(_Body):
    value: int
    size: int = 1   # 1 or 2. Arm checks it.


# --- app factory -------------------------------------------------------------

def default_bus_factory(config: Config) -> BusFactory:
    def factory(arm_cfg: ArmConfig) -> BusProtocol:
        ids = [j.id for j in arm_cfg.joints.values()]
        if config.server.fake:
            from .fake import FakeBus
            return FakeBus(arm_cfg.port, arm_cfg.baudrate, ids=ids)
        from .bus import Bus  # Import only here: fake mode does not need the SDK.
        return Bus(arm_cfg.port, arm_cfg.baudrate)
    return factory


def _validation_detail(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts) or "invalid request"


def create_app(config: Config, bus_factory: BusFactory | None = None, *,
               retry_interval: float = 2.0) -> FastAPI:
    factory = bus_factory or default_bus_factory(config)
    arms: dict[str, Arm] = {
        name: Arm(name, acfg, factory(acfg), poll_hz=config.server.poll_hz,
                  retry_interval=retry_interval, admin=config.server.admin)
        for name, acfg in config.arms.items()
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for arm in arms.values():
            await run_in_threadpool(arm.start)
            log.info("arm %s: status %s", arm.name, arm.status)
        try:
            yield
        finally:
            for arm in arms.values():
                await run_in_threadpool(arm.shutdown)

    app = FastAPI(title="armd", version=__version__,
                  description="HTTP API for the SO-101 arms (Feetech STS3215 servos).",
                  lifespan=lifespan)
    app.state.config = config
    app.state.arms = arms

    # --- error handlers ------------------------------------------------------

    @app.exception_handler(ArmApiError)
    async def _arm_error(request: Request, exc: ArmApiError):
        return JSONResponse(status_code=exc.http_status,
                            content={"error": exc.code, "detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"error": "invalid_request",
                                                      "detail": _validation_detail(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed"}
        return JSONResponse(status_code=exc.status_code,
                            content={"error": codes.get(exc.status_code, "http_error"),
                                     "detail": str(exc.detail)})

    @app.exception_handler(Exception)
    async def _internal_error(request: Request, exc: Exception):
        log.exception("internal error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error",
                                                      "detail": f"{type(exc).__name__}: {exc}"})

    def get_arm(name: str) -> Arm:
        arm = arms.get(name)
        if arm is None:
            raise NotFound(f"no arm {name!r}. Arms: {', '.join(arms) or 'none'}")
        return arm

    # --- endpoints -----------------------------------------------------------

    @app.get("/health")
    def health():
        info = {name: arm.health() for name, arm in arms.items()}
        ok = all(a["status"] == "online" for a in info.values())
        return {"status": "ok" if ok else "degraded", "version": __version__,
                "admin": config.server.admin, "arms": info}

    @app.get("/arms")
    def list_arms():
        return [arm.info() for arm in arms.values()]

    @app.get("/arms/{arm}")
    def arm_info(arm: str):
        return get_arm(arm).info()

    @app.get("/arms/{arm}/state")
    def arm_state(arm: str, fresh: bool = False):
        return get_arm(arm).state(fresh=fresh)

    @app.get("/arms/{arm}/joints/{joint}")
    def joint_state(arm: str, joint: str, fresh: bool = False):
        return get_arm(arm).joint_state(joint, fresh=fresh)

    @app.post("/arms/{arm}/joints/{joint}/position")
    def move_joint(arm: str, joint: str, body: PositionBody):
        return get_arm(arm).move(joint, **body.model_dump())

    @app.post("/arms/{arm}/positions")
    def move_joints(arm: str, body: PositionsBody):
        joints = {name: t.model_dump() for name, t in body.joints.items()}
        return get_arm(arm).move_many(joints, speed=body.speed, acc=body.acc, wait=body.wait,
                                      timeout=body.timeout, tolerance=body.tolerance)

    @app.put("/arms/{arm}/joints/{joint}/speed")
    def set_speed(arm: str, joint: str, body: SpeedBody):
        return get_arm(arm).set_speed(joint, body.speed, body.acc)

    @app.post("/arms/{arm}/joints/{joint}/spin")
    def spin(arm: str, joint: str, body: SpeedBody):
        return get_arm(arm).spin(joint, body.speed, body.acc)

    @app.post("/arms/{arm}/torque")
    def torque(arm: str, body: TorqueBody):
        return get_arm(arm).torque(body.enabled, body.joints)

    @app.post("/arms/{arm}/joints/{joint}/torque")
    def joint_torque(arm: str, joint: str, body: JointTorqueBody):
        return get_arm(arm).torque(body.enabled, [joint])

    @app.post("/arms/{arm}/stop")
    def stop(arm: str, body: StopBody | None = Body(default=None)):
        return get_arm(arm).stop((body or StopBody()).mode)

    @app.post("/arms/{arm}/release")
    def release(arm: str):
        return get_arm(arm).release()

    @app.post("/arms/{arm}/acquire")
    def acquire(arm: str):
        return get_arm(arm).acquire()

    @app.get("/arms/{arm}/joints/{joint}/registers/{addr}")
    def read_register(arm: str, joint: str, addr: int, size: int = 1):
        return get_arm(arm).read_register(joint, addr, size)

    @app.put("/arms/{arm}/joints/{joint}/registers/{addr}")
    def write_register(arm: str, joint: str, addr: int, body: RegisterWriteBody):
        return get_arm(arm).write_register(joint, addr, body.value, body.size)

    @app.websocket("/arms/{arm}/stream")
    async def stream(ws: WebSocket, arm: str, hz: float = 10.0):
        await ws.accept()
        a = arms.get(arm)
        if a is None:
            await ws.send_json({"error": "not_found", "detail": f"no arm {arm!r}"})
            await ws.close(code=1008)
            return
        if not 1 <= hz <= 50:
            await ws.send_json({"error": "invalid_request", "detail": "hz must be 1-50"})
            await ws.close(code=1008)
            return

        async def wait_for_close():
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return

        closed = asyncio.create_task(wait_for_close())
        try:
            while not closed.done():
                await ws.send_json(await run_in_threadpool(a.state))
                await asyncio.wait({closed}, timeout=1.0 / hz)
        except Exception:  # noqa: BLE001 - the client went away
            pass
        finally:
            closed.cancel()

    return app
