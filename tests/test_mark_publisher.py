"""Synthetic pixels verify wall overlay wiring, not real defect accuracy."""

import runpy
import time

import cv2
import numpy as np
import pytest

from brickcell.studio.inspection import InspectionEvidence

inspect_frame = runpy.run_path("scripts/publish_mark_inspection.py")["inspect_frame"]
locate_red_brick = runpy.run_path("scripts/publish_mark_inspection.py")["locate_red_brick"]


def packet():
    return dict(synthetic=False, serial="plate", frame_id="frame-1", time=time.time())


@pytest.mark.parametrize("marked", [False, True])
def test_real_frame_envelope_with_synthetic_pixels_proposes_pink_only_for_visible_mark(marked):
    rgb = np.full((96, 96, 3), [212, 174, 58], np.uint8)
    if marked:
        cv2.line(rgb, (30, 55), (65, 57), (24, 22, 20), 2)
    body = inspect_frame(rgb, packet(), [0, 0, 96, 96], "plate")
    assert ("pink plate" in body["reason"]) is marked
    assert body["decision"] == "UNCALIBRATED"
    assert body["automatic_sorting_allowed"] is False
    assert body["threshold_version"] is None
    assert body["image_jpeg"] != body["overlay_jpeg"]
    store = InspectionEvidence()
    store.publish(body)
    assert store.snapshot()["cameras"]["plate"]["fresh"]


@pytest.mark.parametrize("problem", ["stale", "future", "wrong_serial", "synthetic", "no_id"])
def test_untrusted_camera_frames_cannot_publish_routing_preview(problem):
    p = packet()
    if problem == "stale":
        p["time"] -= 10
    elif problem == "future":
        p["time"] += 10
    elif problem == "wrong_serial":
        p["serial"] = "other"
    elif problem == "synthetic":
        p["synthetic"] = True
    else:
        p["frame_id"] = ""
    with pytest.raises(ValueError, match="fresh real frame"):
        inspect_frame(np.zeros((96, 96, 3), np.uint8), p, [0, 0, 96, 96], "plate")


def test_crop_outside_frame_is_rejected_instead_of_a_clean_result():
    with pytest.raises(ValueError, match="in-frame"):
        inspect_frame(np.zeros((96, 96, 3), np.uint8), packet(), [0, 0, 200, 200], "plate")


def test_tracker_follows_red_brick_and_rejects_absent_or_ambiguous_target():
    image = np.full((240, 320, 3), 120, np.uint8)
    assert locate_red_brick(image) is None
    image[100:140, 180:230] = [220, 30, 30]
    crop = locate_red_brick(image)
    assert crop[0] < 180 < 230 < crop[2]
    assert crop[1] < 100 < 140 < crop[3]
    image[30:70, 30:80] = [220, 30, 30]
    assert locate_red_brick(image) is None


def test_missing_target_never_publishes_no_mark_or_a_route():
    body = inspect_frame(np.zeros((96, 96, 3), np.uint8), packet(), None, "plate")
    assert body["decision"] == "REINSPECT"
    assert body["raw_distance"] is None and body["overlay_jpeg"] is None
    InspectionEvidence().publish(body)
