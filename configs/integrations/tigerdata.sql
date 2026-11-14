-- Apply explicitly to your Tiger Data database; the adapter never changes schema.
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE TABLE IF NOT EXISTS brickcell_inspections (
  case_id TEXT NOT NULL,
  captured_at TIMESTAMPTZ NOT NULL,
  object_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  camera_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  threshold_version TEXT,
  decision TEXT NOT NULL CHECK (decision IN ('GOOD','DEFECTIVE','REINSPECT','ERROR')),
  raw_distance DOUBLE PRECISION,
  latency_ms DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
  synthetic BOOLEAN NOT NULL,
  quality JSONB NOT NULL DEFAULT '{}',
  PRIMARY KEY (case_id, captured_at)
);
SELECT create_hypertable('brickcell_inspections', 'captured_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS brickcell_camera_time
  ON brickcell_inspections (camera_id, model_version, captured_at DESC);
CREATE OR REPLACE VIEW brickcell_hourly AS
SELECT time_bucket('1 hour', captured_at) AS hour, camera_id, model_version,
       threshold_version, count(*) AS inspections,
       avg((decision = 'REINSPECT')::int) AS reinspection_rate,
       avg((decision = 'ERROR')::int) AS error_rate,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS latency_p95_ms
FROM brickcell_inspections
WHERE NOT synthetic
GROUP BY hour, camera_id, model_version, threshold_version;
-- Quality example: SELECT captured_at, camera_id, quality FROM brickcell_inspections
-- WHERE NOT synthetic ORDER BY captured_at DESC LIMIT 100;
