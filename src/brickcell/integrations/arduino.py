"""UNO Q Router Bridge request gate. A button never authorizes robot motion."""

import hashlib
import json
import time
from pathlib import Path

from brickcell.integrations.core import canonical


class ButtonGate:
    def __init__(self):
        self.last_sequence = None

    def poll(self, sequence, context, now=None):
        now = time.time() if now is None else now
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Invalid MCU request counter")
        previous, self.last_sequence = self.last_sequence, sequence
        # On connect/reboot, establish a baseline; never replay old button presses.
        if previous is None or sequence <= previous:
            return None
        if any(
            context.get(k) is not True
            for k in ("camera_ready", "object_present", "operator_confirmed")
        ):
            return None
        captured = context.get("updated_at")
        if type(captured) not in (int, float) or not 0 <= now - captured <= 3:
            return None
        if not all(
            isinstance(context.get(k), str) and context[k].strip()
            for k in ("object_id", "session_id", "camera_id")
        ):
            return None
        event = {k: context[k] for k in ("object_id", "session_id", "camera_id")}
        event.update(
            type="capture.requested",
            source="Arduino UNO Q",
            sequence=sequence,
            requested_at=now,
            robot_authorized=False,
            decision=None,
        )
        event["request_id"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        return event


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/capture-requests"))
    args = parser.parse_args()
    # Imported only on the UNO Q Linux side, never required by the Mac core.
    from arduino.app_utils import App, Bridge

    gate = ButtonGate()
    args.output.mkdir(parents=True, exist_ok=True)

    def poll():
        try:
            context = json.loads(args.context.read_text())
            sequence = Bridge.call("brickcell_take_request")
            event = gate.poll(sequence, context)
            fresh = (
                type(context.get("updated_at")) in (int, float)
                and 0 <= time.time() - context["updated_at"] <= 3
            )
            ready = fresh and all(
                context.get(k) is True
                for k in ("camera_ready", "object_present", "operator_confirmed")
            )
            Bridge.call("brickcell_status", 1 if ready else 0)
            if event:
                path = args.output / (event["request_id"] + ".json")
                with path.open("x") as stream:
                    json.dump(event, stream, indent=2)
        except Exception:
            # Never convert a bridge/context error into a capture or trusted result.
            try:
                Bridge.call("brickcell_status", 2)
            except Exception:
                pass
        time.sleep(0.1)

    App.run(user_loop=poll)


if __name__ == "__main__":
    main()
