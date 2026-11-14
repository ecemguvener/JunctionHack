import cv2
import numpy as np
import pytest

from brickcell.sources.camera import OpenCVCamera, crop_rgb, identifier


def test_camera_crop_preserves_rgb_channels():
    bgr = np.zeros((32, 48, 3), dtype=np.uint8)
    bgr[:, :, 2] = 200
    rgb = crop_rgb(bgr, [8, 4, 32, 28])
    assert rgb.shape == (24, 24, 3)
    assert rgb[0, 0].tolist() == [200, 0, 0]
    with pytest.raises(ValueError, match="outside"):
        crop_rgb(bgr, [0, 0, 100, 100])


def test_failed_capture_releases_camera(monkeypatch):
    class Device:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            self.released = True

    device = Device()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: device)
    with pytest.raises(RuntimeError, match="no frame"):
        with OpenCVCamera("0"):
            pytest.fail("Missing camera frame must stop capture")
    assert device.released


def test_capture_ids_cannot_escape_dataset():
    import argparse

    for value in ("../outside", "/tmp/brick", "..", "a/b"):
        with pytest.raises(argparse.ArgumentTypeError):
            identifier(value)


def test_capture_saves_real_rgb_and_importable_identity(monkeypatch, tmp_path, capsys):
    import json
    from contextlib import contextmanager
    from pathlib import Path

    from PIL import Image

    from brickcell.data.manifest import import_directory, load_manifest
    from brickcell.sources import camera

    @contextmanager
    def fake_camera(*args):
        class Device:
            def read(self):
                bgr = np.zeros((32, 32, 3), dtype=np.uint8)
                bgr[8:24, 8:24, 2] = 255
                return bgr

        yield Device()

    monkeypatch.setattr(camera, "OpenCVCamera", fake_camera)
    root = tmp_path / "raw"
    assert (
        camera.main(
            [
                "--source",
                "0",
                "--label",
                "good",
                "--session",
                "s1",
                "--object-id",
                "brick-1",
                "--output",
                str(root),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["inspection"] is None
    assert report["synthetic"] is False
    assert Image.open(report["image"]).getpixel((10, 10)) == (255, 0, 0)
    assert Path(report["image"]).with_suffix(".json").exists()
    manifest_path = tmp_path / "manifest.json"
    import_directory(root, manifest_path)
    samples = load_manifest(manifest_path).samples
    assert len(samples) == 1
    assert (samples[0].object_id, samples[0].session_id) == ("brick-1", "s1")
