"""MongoDB Atlas: immutable cases and separate, append-only operator reviews."""

import os
import time
import uuid

from brickcell.integrations.core import case_id, required


def store_case(collection, case):
    identifier = case_id(case)
    collection.update_one({"_id": identifier}, {"$setOnInsert": {"inspection": case}}, upsert=True)
    return identifier


def add_review(collection, identifier, reviewer, label, note):
    if label not in ("good", "defective", "uncertain") or not reviewer.strip() or len(note) > 4000:
        raise ValueError("Review requires reviewer, supported human label and bounded note")
    # A separate collection preserves detector output and avoids rewriting ground truth.
    review = {
        "_id": uuid.uuid4().hex,
        "case_id": identifier,
        "reviewer": reviewer,
        "human_label": label,
        "note": note,
        "reviewed_at": time.time(),
    }
    collection.insert_one(review)
    return review["_id"]


def run(case, context):
    from pymongo import MongoClient

    with MongoClient(
        required("MONGODB_URI"),
        tls=True,
        tlsAllowInvalidCertificates=False,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    ) as client:
        database = client[os.environ.get("MONGODB_DATABASE", "brickcell")]
        identifier = store_case(database["inspection_cases"], case)
    return {
        "case_id": identifier,
        "collection": "inspection_cases",
        "human_review_collection": "operator_reviews",
        "detector_output_immutable": True,
    }
