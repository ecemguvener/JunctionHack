"""Small dependency-free checks shared by simulator endpoints and offline tests."""


def accept_inspection(capture: dict | None, result: dict) -> dict | None:
    if not capture or any(
        result.get(key) != capture.get(key)
        for key in ("object_id", "frame_id", "timestamp", "view_id")
    ):
        raise ValueError("Stale or mismatched camera result")
    if result.get("inspector_mode") != "anomalib_openvino_cpu":
        raise ValueError("Only actual detector results are accepted")
    return result if result.get("is_valid_view") else None


def authorize_sort(result: dict | None, operation: str, object_id: str) -> None:
    expected = {"move_good": "GOOD", "move_defective": "DEFECTIVE"}.get(operation)
    if (
        expected is None
        or not result
        or not result.get("is_valid_view")
        or result.get("object_id") != object_id
        or result.get("decision") != expected
    ):
        raise ValueError("Bowl motion requires this object's published camera decision")
