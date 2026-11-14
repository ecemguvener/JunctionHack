# Aiven for Apache Kafka — inspection event delivery

Aiven's managed Kafka receives versioned `capture.completed`,
`inspection.completed` and `inspection.needs_review` events after local
inspection. Default event type derives from the recorded decision; an explicit
case `event_type` can represent a completed capture. Consumers can update case
storage, analytics or UI independently. Nothing waits for Kafka inside a robot
cycle. Event keys preserve object/session grouping and `event_id` is the immutable
case hash. Consumers must deduplicate that ID across producer restarts.

Enable SASL/SCRAM authentication on an Aiven Kafka service and configure
`AIVEN_KAFKA_BOOTSTRAP`, `AIVEN_KAFKA_USER`, `AIVEN_KAFKA_PASSWORD`,
`AIVEN_KAFKA_TOPIC`, optionally `AIVEN_KAFKA_CA` for its CA file. Create the topic
and writer ACL before running. The connector does not provision services.

```bash
uv run --with 'confluent-kafka>=2.6,<3' python -m brickcell.integrations \
  --case artifacts/case.json --providers aiven --execute
```

The optional client uses SASL_SSL, idempotent production and all-replica acks.
Success requires a delivery callback with partition/offset before the bounded
flush deadline. Failed or unacknowledged delivery remains retryable in the local
outbox. A partition offset is delivery evidence, not a physical inspection result.
No image bytes or credentials are embedded in events; cases include local
relative asset references/hashes. Secure consumers and retention accordingly.

Offline producer tests verify event identity, TLS and callback/deadline handling.
A live broker, consumer flow and sustained throughput remain unverified.
[Aiven Kafka documentation](https://aiven.io/docs/products/kafka).
