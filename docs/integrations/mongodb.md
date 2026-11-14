# MongoDB Atlas — inspection case history and operator review

Atlas holds each complete inspection case under its content-hash `_id`. Insert-once
upserts preserve the exact recorded model decision, identities, evidence hashes,
quality metadata and reason. Human labels and review notes belong in a separate
`operator_reviews` collection, linked by case ID; they never overwrite predictions.
This is the document case store, while Tiger Data serves time-series queries.
Either can be selected independently in the sidecar.

Create an Atlas database/user and network access policy. Export `MONGODB_URI`
and optionally `MONGODB_DATABASE` (default `brickcell`), then:

```bash
uv run --with 'pymongo>=4.10,<5' python -m brickcell.integrations \
  --case artifacts/case.json --providers mongodb --execute
```

TLS verification and connection/server-selection/socket deadlines are enforced.
MongoDB's `_id` unique index deduplicates cases without a migration. For operator
review, call `add_review(database['operator_reviews'], case_id, reviewer, label,
note)` from an authenticated operator workflow. The API accepts good/defective/
uncertain human labels with a bounded note and assigns its own review ID/time.
Do not call a human label the model's output or use the same brick in different
train/test partitions. A sensible additional index is
`operator_reviews.create_index([('case_id', 1), ('reviewed_at', -1)])`.

Use `inspection_cases.find({'inspection.object_id': ...})` to review a brick's
history. Atlas Vector Search is future work: this feature does not invent
embeddings, similarity scores or a vector index. Offline tests verify immutable
upserts and separate human reviews; live Atlas access remains pending.
[PyMongo CRUD documentation](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/).
