from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# YOLO labels are written with 6 decimals, so an edge can land 5e-7 outside the
# frame purely from rounding. Only report a box as clipped past this tolerance.
CLIP_TOL = 1e-4


def iter_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


@dataclass
class BoxRecord:
    """One YOLO bbox, normalised cx/cy/w/h, already clipped into [0, 1]."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    line_no: int
    clipped: bool = False


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path
    boxes: list[BoxRecord] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.image_path.stem


@dataclass
class DiscoveryReport:
    """Everything the validation pass found, so EDA and the write-up can cite it."""

    n_images: int = 0
    n_labels: int = 0
    n_boxes: int = 0
    images_without_label: list[str] = field(default_factory=list)
    labels_without_image: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    malformed_lines: list[tuple[str, int, str]] = field(default_factory=list)
    clipped_boxes: list[tuple[str, int]] = field(default_factory=list)
    dropped_boxes: list[tuple[str, int, str]] = field(default_factory=list)
    provided_splits: dict[str, int] = field(default_factory=dict)
    images_outside_provided_splits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "n_labels": self.n_labels,
            "n_boxes": self.n_boxes,
            "n_images_without_label": len(self.images_without_label),
            "n_labels_without_image": len(self.labels_without_image),
            "n_empty_labels": len(self.empty_labels),
            "n_malformed_lines": len(self.malformed_lines),
            "n_clipped_boxes": len(self.clipped_boxes),
            "n_dropped_boxes": len(self.dropped_boxes),
            "images_without_label": self.images_without_label[:50],
            "labels_without_image": self.labels_without_image[:50],
            "empty_labels": self.empty_labels,
            "malformed_lines": [list(x) for x in self.malformed_lines[:50]],
            "clipped_boxes": [list(x) for x in self.clipped_boxes[:50]],
            "dropped_boxes": [list(x) for x in self.dropped_boxes[:50]],
            "provided_splits": self.provided_splits,
            "n_images_outside_provided_splits": len(self.images_outside_provided_splits),
            "images_outside_provided_splits": self.images_outside_provided_splits,
        }


def find_dataset_root(
    root: str | Path, images_dirname: str = "images", labels_dirname: str = "labels"
) -> Path:
    """Locate the folder that actually holds images/ + labels/.

    Kaggle mounts the set at /kaggle/input/<slug>/ but the archive may add one or
    two wrapper folders (e.g. .../vietnamese-traffic-signs/dataset/images). Search
    a few levels down rather than making the caller get the path exactly right.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    candidates = [root, *sorted(p for p in root.rglob("*") if p.is_dir())]
    for base in candidates:
        if (base / images_dirname).is_dir() and (base / labels_dirname).is_dir():
            return base
    raise FileNotFoundError(
        f"no folder under {root} contains both '{images_dirname}/' and '{labels_dirname}/'"
    )


def _parse_label_file(
    path: Path, stem: str, num_classes: int, report: DiscoveryReport
) -> list[BoxRecord]:
    boxes: list[BoxRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    if not any(line.strip() for line in raw_lines):
        report.empty_labels.append(stem)
        return boxes
    for i, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            report.malformed_lines.append((stem, i, f"expected 5 fields, got {len(parts)}"))
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            report.malformed_lines.append((stem, i, "non-numeric field"))
            continue
        if not 0 <= class_id < num_classes:
            report.malformed_lines.append((stem, i, f"class id {class_id} out of range"))
            continue
        if w <= 0 or h <= 0:
            report.dropped_boxes.append((stem, i, "zero or negative size"))
            continue

        # Clip to the frame: VNTS has boxes whose edge runs past the border.
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(1.0, x2), min(1.0, y2)
        clipped = max(-x1, -y1, x2 - 1.0, y2 - 1.0) > CLIP_TOL
        if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
            report.dropped_boxes.append((stem, i, "box lies entirely outside the frame"))
            continue
        if clipped:
            report.clipped_boxes.append((stem, i))
        boxes.append(
            BoxRecord(
                class_id=class_id,
                cx=(cx1 + cx2) / 2,
                cy=(cy1 + cy2) / 2,
                w=cx2 - cx1,
                h=cy2 - cy1,
                line_no=i,
                clipped=clipped,
            )
        )
    return boxes


def discover_yolo_dataset(
    root: str | Path,
    num_classes: int,
    images_dirname: str = "images",
    labels_dirname: str = "labels",
) -> tuple[list[ImageRecord], DiscoveryReport]:
    """Read a YOLO detection layout and validate it. Returns only usable records."""
    base = find_dataset_root(root, images_dirname, labels_dirname)
    img_dir, lbl_dir = base / images_dirname, base / labels_dirname

    images = {p.stem: p for p in iter_images(img_dir)}
    labels = {p.stem: p for p in sorted(lbl_dir.glob("*.txt"))}

    report = DiscoveryReport(n_images=len(images), n_labels=len(labels))
    report.images_without_label = sorted(set(images) - set(labels))
    report.labels_without_image = sorted(set(labels) - set(images))

    records: list[ImageRecord] = []
    for stem in sorted(set(images) & set(labels)):
        boxes = _parse_label_file(labels[stem], stem, num_classes, report)
        report.n_boxes += len(boxes)
        records.append(ImageRecord(image_path=images[stem], label_path=labels[stem], boxes=boxes))

    _read_provided_splits(base, set(images), report)
    return records, report


def _read_provided_splits(base: Path, image_stems: set[str], report: DiscoveryReport) -> None:
    """Record the authors' train/test lists. We do not split on them (they leak
    near-duplicate frames), but the counts belong in the report."""
    split_dir = base / "split_dataset"
    if not split_dir.is_dir():
        return
    listed: set[str] = set()
    for f in sorted(split_dir.glob("*_files.txt")):
        with open(f, "r", encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        report.provided_splits[f.stem.replace("_files", "")] = len(names)
        listed |= {Path(n).stem for n in names}
    report.images_outside_provided_splits = sorted(image_stems - listed)


def iter_boxes(records: list[ImageRecord]) -> Iterator[tuple[ImageRecord, BoxRecord]]:
    for rec in records:
        for box in rec.boxes:
            yield rec, box


def discover_classes(root: str | Path) -> list[str]:
    """Folder-per-class layouts only. VNTS is not one of them: its class list
    comes from configs/classes.csv via vn_tsc.data.classes.ClassTable."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
