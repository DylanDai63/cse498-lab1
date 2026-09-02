# Step 2 — Fine-tuning dataset

50 images / 2 classes: `tomato` (34), `toilet_paper` (16) — the two failure cases from Step 1 chosen as fine-tuning targets (tomato was detected as apple, toilet paper as cup/toilet).

**Source**: subsampled from Google Open Images (validation + test splits, bbox annotations), classes `/m/07j87` (Tomato) and `/m/09gtd` (Toilet paper). Group/depiction boxes excluded; images with 1-4 target boxes preferred; resized to max 640px; converted to YOLO format; split train 42 / valid 8 (85/15). Reproducible via `build_dataset.py`. The assignment-suggested Kaggle grocery dataset was verified to contain only tabular product data (no images), so this dataset was created per the assignment's "Find/Create" instruction.

Open Images images are sourced from Flickr under CC BY 2.0; annotations CC BY 4.0 by Google.

- `dataset/` — YOLO layout + `data.yaml` (nc=2)
- `dataset.zip` — same content, for direct upload into the Colab runtime

## Compliance with the revised Lab 1 instructions (instructor email, 2026-09-01)

- **Bounding-box ground truth**: every image carries YOLO-format box labels sourced from Open Images official bbox annotations. ✓
- **Unseen/new classes for the pretrained model**: verified programmatically against `yolo11n.pt` class list — neither `tomato` nor `toilet paper` is among the 80 COCO classes (which also motivated dropping `bowl`, a COCO class, from an earlier 3-class draft). ✓
- **YOLO format, directly usable in Colab**: `dataset/` follows the standard YOLO layout (`train/valid` x `images/labels` + `data.yaml`, nc=2); `dataset.zip` uploads straight into the Colab runtime. ✓

Sourcing approach: images collected from the web (Google Open Images) that already carry bounding-box annotations — consistent with the revised instructions, which accept web-collected images and existing annotated datasets; Roboflow Annotate was unnecessary since box labels already existed.
