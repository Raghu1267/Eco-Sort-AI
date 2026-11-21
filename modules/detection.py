"""YOLOv8 detection helpers (no Streamlit imports, so they are easy to test).

All detections come from real inference with the trained model in model/best.pt.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "best.pt"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_FILE_MB = 15
MAX_SIDE_PX = 3000  # larger images are scaled down (keeps memory use reasonable on CPU)
INFERENCE_SIZE = 640  # input size the model was trained with

DETECTION_COLUMNS = ["Detection ID", "Class", "Confidence (%)", "X1", "Y1", "X2", "Y2"]

# Box colours (RGB), assigned per class id so a class always has the same colour.
_PALETTE = [
    (61, 220, 151), (45, 212, 191), (96, 165, 250), (250, 204, 21), (251, 146, 60),
    (244, 114, 182), (167, 139, 250), (248, 113, 113), (132, 204, 22), (34, 211, 238),
    (232, 121, 249), (253, 224, 71),
]


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable JPG/PNG image."""


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


@dataclass
class DetectionResult:
    annotated_image: Image.Image
    detections: pd.DataFrame
    inference_ms: float
    total_ms: float
    image_size: Tuple[int, int]
    conf_threshold: float
    iou_threshold: float
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(model_path: Path = MODEL_PATH):
    """Load the trained YOLOv8s model on CPU. Wrap with st.cache_resource in the app."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelLoadError(
            f"Model file not found at '{model_path}'. Place your trained weights at model/best.pt."
        )
    try:
        from ultralytics import YOLO  # imported lazily so the UI can show a clear error
    except ImportError as exc:  # pragma: no cover
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        model = YOLO(str(model_path))
    except Exception as exc:  # corrupted file, version mismatch, etc.
        raise ModelLoadError(f"Could not load '{model_path.name}': {exc}") from exc
    return model


def get_class_names(model) -> Dict[int, str]:
    """Return the class-id to class-name mapping stored in the trained model."""
    names = model.names
    if isinstance(names, (list, tuple)):
        return {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def get_model_info(model) -> Dict[str, Any]:
    """Collect basic facts about the loaded model (read from the model itself)."""
    names = get_class_names(model)
    info: Dict[str, Any] = {"num_classes": len(names), "class_names": names}
    try:
        train_args = getattr(model, "ckpt", {}).get("train_args", {}) or {}
        info["train_imgsz"] = train_args.get("imgsz")
        info["train_epochs"] = train_args.get("epochs")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Image validation
# --------------------------------------------------------------------------- #
def validate_image(data: bytes, filename: str) -> Image.Image:
    """Validate an uploaded file and return an upright RGB PIL image.

    Raises ImageValidationError with a user-friendly message on any problem.
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a JPG, JPEG or PNG image."
        )
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ImageValidationError(f"The file is larger than {MAX_FILE_MB} MB. Please upload a smaller image.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated/corrupt files
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("This file could not be read as an image. It may be corrupted.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"The file content is {fmt or 'unknown'}, not JPEG or PNG. Please upload a real JPG, JPEG or PNG image."
        )
    img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    if min(img.size) < 32:
        raise ImageValidationError("The image is too small (minimum 32 x 32 pixels).")
    if max(img.size) > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return img


# --------------------------------------------------------------------------- #
# Inference and drawing
# --------------------------------------------------------------------------- #
def _empty_detections() -> pd.DataFrame:
    return pd.DataFrame(columns=DETECTION_COLUMNS)


def draw_detections(image: Image.Image, detections: pd.DataFrame) -> Image.Image:
    """Draw boxes with 'Class confidence%' labels using the model's own outputs."""
    canvas = np.array(image.convert("RGB"))
    h, w = canvas.shape[:2]
    scale = max(h, w) / 1000.0
    box_thickness = max(2, int(round(scale * 3)))
    font_scale = max(0.5, scale * 0.7)
    text_thickness = max(1, int(round(scale * 1.6)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = (int(row["X1"]), int(row["Y1"]), int(row["X2"]), int(row["Y2"]))
        color = _PALETTE[hash_class(row["Class"]) % len(_PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        label = f'{row["Class"]} {row["Confidence (%)"]:.1f}%'
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        pad = max(2, box_thickness)
        label_h = th + baseline + pad * 2
        top = y1 - label_h
        if top < 0:  # not enough room above the box: put the label inside it
            top = y1
        left = min(max(0, x1), max(0, w - tw - pad * 2))
        cv2.rectangle(canvas, (left, top), (left + tw + pad * 2, top + label_h), color, -1)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (10, 20, 30) if luminance > 140 else (255, 255, 255)
        cv2.putText(canvas, label, (left + pad, top + th + pad), font, font_scale, text_color,
                    text_thickness, cv2.LINE_AA)
    return Image.fromarray(canvas)


def hash_class(name: str) -> int:
    """Stable (process-independent) integer for a class name."""
    return sum((i + 1) * ord(c) for i, c in enumerate(str(name)))


def run_detection(model, image: Image.Image, conf: float = 0.25, iou: float = 0.45) -> DetectionResult:
    """Run real YOLOv8 inference on a PIL image and return the annotated result."""
    start = time.perf_counter()
    rgb = image.convert("RGB")
    bgr = np.ascontiguousarray(np.array(rgb)[:, :, ::-1])  # ultralytics expects BGR for ndarrays

    results = model.predict(
        source=bgr,
        conf=float(conf),
        iou=float(iou),
        imgsz=INFERENCE_SIZE,
        device="cpu",
        verbose=False,
    )
    res = results[0]
    names = res.names
    boxes = res.boxes

    rows = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-confs)  # highest confidence first
        for rank, idx in enumerate(order, start=1):
            x1, y1, x2, y2 = xyxy[idx]
            rows.append(
                {
                    "Detection ID": rank,
                    "Class": str(names[int(classes[idx])]),
                    "Confidence (%)": round(float(confs[idx]) * 100.0, 2),
                    "X1": int(round(x1)),
                    "Y1": int(round(y1)),
                    "X2": int(round(x2)),
                    "Y2": int(round(y2)),
                }
            )
    detections = pd.DataFrame(rows, columns=DETECTION_COLUMNS) if rows else _empty_detections()
    annotated = draw_detections(rgb, detections) if rows else rgb

    speed = getattr(res, "speed", None) or {}
    return DetectionResult(
        annotated_image=annotated,
        detections=detections,
        inference_ms=float(speed.get("inference", 0.0)),
        total_ms=(time.perf_counter() - start) * 1000.0,
        image_size=rgb.size,
        conf_threshold=float(conf),
        iou_threshold=float(iou),
    )


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
