"""Aiven for Apache Kafka: asynchronous inspection events with delivery acknowledgement."""

import os

from brickcell.integrations.core import canonical, case_id, required


def event(case):
    kind = case.get("event_type") or (
        "inspection.needs_review"
        if case["decision"] in ("REINSPECT", "ERROR")
        else "inspection.completed"
    )
    if kind not in ("capture.completed", "inspection.completed", "inspection.needs_review"):
        raise ValueError("Unknown inspection event type")
    return {
        "schema_version": 1,
        "event_id": case_id(case),
        "type": kind,
        "object_id": case["object_id"],
        "session_id": case["session_id"],
        "captured_at": case["captured_at"],
        "inspection": case,
    }


def run(case, context):
    from confluent_kafka import Producer

    settings = {
        "bootstrap.servers": required("AIVEN_KAFKA_BOOTSTRAP"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "SCRAM-SHA-256",
        "sasl.username": required("AIVEN_KAFKA_USER"),
        "sasl.password": required("AIVEN_KAFKA_PASSWORD"),
        "enable.idempotence": True,
        "acks": "all",
        "message.timeout.ms": 10000,
        "socket.timeout.ms": 5000,
    }
    if os.environ.get("AIVEN_KAFKA_CA"):
        settings["ssl.ca.location"] = os.environ["AIVEN_KAFKA_CA"]
    producer = Producer(settings)
    acknowledgements = []

    def delivered(error, message):
        if error:
            acknowledgements.append(None)
        else:
            acknowledgements.append({"partition": message.partition(), "offset": message.offset()})

    topic = required("AIVEN_KAFKA_TOPIC")
    producer.produce(
        topic,
        key=canonical([case["object_id"], case["session_id"]]).encode(),
        value=canonical(event(case)).encode(),
        on_delivery=delivered,
    )
    remaining = producer.flush(12)
    if remaining or len(acknowledgements) != 1 or acknowledgements[0] is None:
        raise RuntimeError("Kafka event was not acknowledged before deadline")
    return {
        "topic": topic,
        "event_id": case_id(case),
        **acknowledgements[0],
        "consumer_deduplication_key": "event_id",
    }
