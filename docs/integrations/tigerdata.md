# Tiger Data — inspection time-series analytics

Tiger Data stores acquisition time, object/session/camera identities, detector
and threshold versions, raw anomaly distance, latency, synthetic status and camera
quality JSON. This supports reinspection/error trends and camera-quality drift
without mixing scores from different detector models. SQL uses parameterized
inserts and a case/time key, making redelivery idempotent. No schema migration is
performed implicitly during inspection.

Create a Tiger Data PostgreSQL service, export `TIGERDATA_URL` and configure its
trusted root certificate (`PGSSLROOTCERT` if required). Apply
`configs/integrations/tigerdata.sql` using psql, then run:

```bash
uv run --with 'psycopg[binary]>=3.2,<4' python -m brickcell.integrations \
  --case artifacts/case.json --providers tigerdata --execute
```

The extra driver is optional and does not change core dependencies. Connections
require TLS hostname verification and bounded connection/statement waits. Query
`brickcell_hourly` for counts, reinspection/error rates and p95 latency by
camera/model/threshold. Synthetic demo rows are excluded from that view.
Missing scores remain SQL NULL, not fabricated zero distances. Put actual focus,
blur or depth-quality measurements in the case's `quality` field; metrics are
not invented by the connector. Use a least-privilege writer after migration.

Offline tests verify parameterization, idempotency, TLS and acquisition time.
A real database migration, Timescale queries and trend charts remain pending.
[Tiger Data documentation](https://www.tigerdata.com/docs/build).
