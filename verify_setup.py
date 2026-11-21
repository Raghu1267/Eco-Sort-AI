"""Quick self-check: loads model/best.pt, checks the class mapping and runs one real inference.

Usage:
    python verify_setup.py                 # runs on a blank test image (expects 0 detections)
    python verify_setup.py path/to/photo.jpg
"""
import sys
import time

from PIL import Image

from modules.detection import MODEL_PATH, get_class_names, get_model_info, load_model, run_detection, validate_image
from modules.segregation import add_segregation, check_mapping


def main() -> int:
    print(f"Model file: {MODEL_PATH}")
    t0 = time.perf_counter()
    model = load_model()
    print(f"[ok] Model loaded in {time.perf_counter() - t0:.1f}s")

    names = get_class_names(model)
    info = get_model_info(model)
    print(f"[ok] {len(names)} classes in the model (train imgsz={info.get('train_imgsz')}, epochs={info.get('train_epochs')})")
    mapping = check_mapping(names.values())
    if mapping["unmapped"] or mapping["unused"]:
        print("[!!] Class mismatch. Unmapped model classes:", mapping["unmapped"], "| unused rules:", mapping["unused"])
        return 1
    print("[ok] Every model class has a segregation rule (names match exactly)")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as fh:
            image = validate_image(fh.read(), sys.argv[1])
    else:
        image = Image.new("RGB", (640, 480), (128, 128, 128))
    result = run_detection(model, image, conf=0.25)
    print(f"[ok] CPU inference finished in {result.total_ms:.0f} ms, {len(result.detections)} detection(s)")
    if len(result.detections):
        print(add_segregation(result.detections)[["Class", "Confidence (%)", "Segregation Bin", "Hazard Priority"]].to_string(index=False))
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
