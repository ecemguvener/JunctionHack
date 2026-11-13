"""Train Anomalib PatchCore on real brick photos; never sends arm commands."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/bricks"))
    parser.add_argument("--output", type=Path, default=Path("inspection/output/patchcore"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--backbone", choices=("wide_resnet50_2", "resnet18"), default="wide_resnet50_2")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    missing = []
    for folder, minimum in (("train/good", 2), ("test/good", 4), ("test/bad", 4)):
        path = args.data / folder
        path.mkdir(parents=True, exist_ok=True)
        count = sum(p.is_file() and p.suffix.lower() in extensions for p in path.rglob("*"))
        print("%s: %d images" % (path, count))
        if count < minimum:
            missing.append("%s needs at least %d images" % (folder, minimum))
    if missing:
        parser.exit(2, "Dataset not ready: " + "; ".join(missing) + "\n")
    if args.check_only:
        return
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as exc:
        parser.exit(2, "Anomalib dependencies unavailable: %s. See inspection/README.md.\n" % exc)
    data = Folder(
        name="bricks", root=args.data.resolve(), normal_dir="train/good",
        normal_test_dir="test/good", abnormal_dir="test/bad",
        train_batch_size=4, eval_batch_size=4, num_workers=0,
        val_split_mode="from_test", val_split_ratio=0.5, seed=42,
    )
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    model = Patchcore(backbone=args.backbone,
                     pre_processor=Patchcore.configure_pre_processor(image_size=(args.image_size, args.image_size)))
    engine = Engine(default_root_dir=str(args.output.resolve()), accelerator="cpu", devices=1)
    engine.fit(model=model, datamodule=data)
    print(engine.test(model=model, datamodule=data))
    print("Model artifacts:", args.output.resolve())


if __name__ == "__main__":
    main()
