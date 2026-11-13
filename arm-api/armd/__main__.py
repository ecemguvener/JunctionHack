"""Run the service: python -m armd [--config PATH] [--fake] [--host H] [--port P]."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import ConfigError, load_config
from .server import create_app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="armd", description="HTTP API for the SO-101 arms.")
    p.add_argument("--config", help="config file (default: env ARMD_CONFIG, else config.yaml)")
    p.add_argument("--fake", action="store_true", help="use simulated servos (no hardware)")
    p.add_argument("--host", help="override server.host")
    p.add_argument("--port", type=int, help="override server.port")
    p.add_argument("--log-level", default="info", help="debug, info, warning, error")
    args = p.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("armd")

    try:
        config = load_config(args.config, fake=args.fake or None)
    except ConfigError as e:
        log.error("%s", e)
        return 2
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    log.info("config %s, fake=%s, admin=%s, arms=%s", config.path, config.server.fake,
             config.server.admin, ", ".join(config.arms) or "none")
    app = create_app(config)
    # log_config=None: uvicorn uses the root logger above (stderr, timestamps).
    uvicorn.run(app, host=config.server.host, port=config.server.port,
                log_config=None, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
