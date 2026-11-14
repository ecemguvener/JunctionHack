from brickcell.integrations.mongodb import add_review, store_case


def test_atlas_keeps_model_output_immutable_and_reviews_separate(image_case):
    calls = []

    class Collection:
        def update_one(self, where, update, upsert):
            calls.append((where, update, upsert))

        def insert_one(self, review):
            calls.append(review)

    cases, reviews = Collection(), Collection()
    identifier = store_case(cases, image_case)
    assert "$setOnInsert" in calls[0][1]
    assert "$set" not in calls[0][1]
    add_review(reviews, identifier, "operator-1", "defective", "Visible dark line")
    assert calls[1]["human_label"] == "defective"
    assert calls[1]["case_id"] == identifier
    assert image_case["decision"] == "REINSPECT"
