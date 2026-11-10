import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


DATA = Path("/kaggle/input/datasets/guzzbr/draw-real-capture-pool-v1")
OUT = Path("/kaggle/working/review_calibration")
FRAME_CSV = DATA / "frames.csv"


def build_file_index():
    index = defaultdict(list)
    for path in DATA.rglob("*"):
        if path.is_file():
            index[path.name].append(path)
    return index


FILE_INDEX = build_file_index()


def resolve_asset(relative_path, image=False):
    relative = Path(relative_path)
    candidates = [DATA / relative]
    if image:
        candidates.append(DATA / "images" / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = FILE_INDEX.get(relative.name, [])
    return matches[0] if len(matches) == 1 else None


def choose_spread(rows, count):
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[index] for index in indices]


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None, []
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    values = np.unique(gray)
    scaled = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    preview = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
    return preview, [int(value) for value in values[:32]]


with FRAME_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

by_session = defaultdict(list)
for row in rows:
    if (
        row.get("pseudo_mask_path_in_archive")
        and row.get("mask_status") == "valid_exploratory_pseudo"
    ):
        by_session[row["session_id"]].append(row)

selected = []
per_session_limit = 16 if len(by_session) == 1 else 2
for session_id in sorted(by_session):
    chosen = choose_spread(by_session[session_id], per_session_limit)
    for row in chosen:
        selected.append((session_id, row))

columns, cell_w, image_h, caption_h = 4, 560, 315, 64
rows_n = (len(selected) + columns - 1) // columns
sheet = np.full(
    (rows_n * (image_h + caption_h), columns * cell_w, 3), 248, dtype=np.uint8
)
review_rows = []
bad_assets = []
class_values = Counter()

for index, (session_id, row) in enumerate(selected):
    image_path = resolve_asset(row["image_path_in_archive"], image=True)
    mask_path = resolve_asset(row["pseudo_mask_path_in_archive"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
    mask_preview, mask_values = load_mask(mask_path) if mask_path else (None, [])
    if image is None or mask_preview is None:
        bad_assets.append(
            {
                "frame_id": row["frame_id"],
                "image_ok": image is not None,
                "mask_ok": mask_preview is not None,
            }
        )
        continue
    x, y = (index % columns) * cell_w, (index // columns) * (image_h + caption_h)
    half = cell_w // 2
    for panel, content in enumerate((image, mask_preview)):
        scale = min(half / content.shape[1], image_h / content.shape[0])
        resized = cv2.resize(
            content,
            (round(content.shape[1] * scale), round(content.shape[0] * scale)),
            interpolation=cv2.INTER_AREA if panel == 0 else cv2.INTER_NEAREST,
        )
        ox = x + panel * half + (half - resized.shape[1]) // 2
        oy = y + (image_h - resized.shape[0]) // 2
        sheet[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    display_session = f"S{sorted(by_session).index(session_id) + 1:02d}"
    title = (
        f"{display_session} | frame {row['frame_index']} | {row['annotation_status']}"
    )
    cv2.putText(
        sheet,
        title[:72],
        (x + 6, y + image_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "left: source  |  right: pseudo-mask preview",
        (x + 6, y + image_h + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    class_values.update(mask_values)
    review_rows.append(
        {
            "frame_id": row["frame_id"],
            "session_id": session_id,
            "frame_index": row["frame_index"],
            "annotation_status": row["annotation_status"],
            "annotation_authority": row["annotation_authority"],
            "training_approved_before_review": row["training_approved"],
            "review_decision": "",
            "review_notes": "",
        }
    )

OUT.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT / "pseudo-mask-review-pilot.jpg"), sheet)
with (OUT / "review-template.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(review_rows[0])
        if review_rows
        else [
            "frame_id",
            "session_id",
            "frame_index",
            "review_decision",
            "review_notes",
        ],
    )
    writer.writeheader()
    writer.writerows(review_rows)

class_names = {
    0: ("background", (54, 54, 54)),
    1: ("paper", (220, 220, 220)),
    2: ("hand", (70, 70, 235)),
    3: ("ink", (240, 90, 45)),
    255: ("ignore", (220, 50, 220)),
}
frame_review_files = []
panel_w, panel_h = 960, 540
for index, (session_id, row) in enumerate(selected):
    image_path = resolve_asset(row["image_path_in_archive"], image=True)
    mask_path = resolve_asset(row["pseudo_mask_path_in_archive"])
    if image_path is None or mask_path is None:
        continue
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or raw_mask is None:
        continue
    mask = (
        cv2.cvtColor(raw_mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        if raw_mask.ndim == 3
        else raw_mask
    )
    if mask.shape != image.shape[:2]:
        continue
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    overlay = image.copy()
    class_counts = {}
    for value, (name, color) in class_names.items():
        pixels = mask == value
        count = int(np.count_nonzero(pixels))
        if count:
            class_counts[name] = count
        color_mask[pixels] = color
        if value not in (0,):
            overlay[pixels] = (
                0.48 * image[pixels] + 0.52 * np.asarray(color, dtype=np.float32)
            ).astype(np.uint8)
    panels = [image, color_mask, overlay]
    canvas = np.full((panel_h + 108, panel_w * 3, 3), 250, dtype=np.uint8)
    heading = f"S{sorted(by_session).index(session_id) + 1:02d} | frame {row['frame_index']} | {row['annotation_status']}"
    cv2.putText(
        canvas,
        heading,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for panel, (content, label) in enumerate(
        zip(panels, ("source", "discrete classes", "overlay"))
    ):
        resized = cv2.resize(
            content,
            (panel_w, panel_h),
            interpolation=cv2.INTER_AREA if panel == 0 else cv2.INTER_NEAREST,
        )
        x = panel * panel_w
        canvas[38 : 38 + panel_h, x : x + panel_w] = resized
        cv2.putText(
            canvas,
            label,
            (x + 8, panel_h + 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    legend = "0 background  1 paper  2 hand  3 ink  255 ignore"
    cv2.putText(
        canvas,
        legend,
        (8, panel_h + 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    output_name = f"frame-review-{index + 1:02d}.jpg"
    cv2.imwrite(str(OUT / output_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    frame_review_files.append(
        {
            "file": output_name,
            "frame_id": row["frame_id"],
            "class_pixel_counts": class_counts,
        }
    )

summary = {
    "source_dataset": "guzzbr/draw-real-capture-pool-v1",
    "source_frames": len(rows),
    "sessions_with_pseudo_masks": len(by_session),
    "pseudo_mask_candidates": sum(len(items) for items in by_session.values()),
    "selected_for_calibration": len(review_rows),
    "selection": "up to two time-spaced proposals per session, or up to sixteen when only one session has candidates; exploratory only",
    "training_approved_by_this_job": 0,
    "mask_values_seen": sorted(class_values),
    "asset_errors": bad_assets,
    "per_session": {
        f"S{i + 1:02d}": len(by_session[sid])
        for i, sid in enumerate(sorted(by_session))
    },
    "outputs": [
        "pseudo-mask-review-pilot.jpg",
        "review-template.csv",
        *[item["file"] for item in frame_review_files],
    ],
    "frame_review_details": frame_review_files,
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
