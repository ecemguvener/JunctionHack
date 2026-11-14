"""Tiger Data / Timescale: acquisition-time inspection telemetry, grouped by model."""

from datetime import UTC, datetime

from brickcell.integrations.core import canonical, case_id, required

INSERT = """
INSERT INTO brickcell_inspections
(case_id,captured_at,object_id,session_id,camera_id,model_version,threshold_version,
 decision,raw_distance,latency_ms,synthetic,quality)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
ON CONFLICT (case_id,captured_at) DO NOTHING
"""


def run(case, context):
    import psycopg

    values = (
        case_id(case),
        datetime.fromtimestamp(case["captured_at"], UTC),
        case["object_id"],
        case["session_id"],
        case["camera_id"],
        case["model_version"],
        case.get("threshold_version"),
        case["decision"],
        case.get("score"),
        case["inference_latency_ms"],
        case["synthetic"],
        canonical(case.get("quality", {})),
    )
    with psycopg.connect(
        required("TIGERDATA_URL"),
        sslmode="verify-full",
        connect_timeout=5,
        options="-c statement_timeout=10000",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(INSERT, values)
    return {
        "case_id": case_id(case),
        "table": "brickcell_inspections",
        "time_basis": "camera_acquisition",
        "idempotent": True,
    }
