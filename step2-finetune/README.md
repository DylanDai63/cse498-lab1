# Step 2 — Fine-tuning dataset

25 curated images / 1 class: `tomato` — the Step 1 failure case chosen as the fine-tuning target (the pretrained model detected tomatoes as apples). Iteration history, kept as lessons for the report: (a) a 2-class draft (tomato+toilet_paper) showed a class with only ~12 training images fails to learn (mAP50 0.01 vs 0.59); (b) a 50-image random subsample trained to mAP50 0.66 but with detection confidences below 0.25 — label noise in Open Images' Tomato class (slices, sauces, mislabeled persimmons, unripe-only shots) diluted the signal; (c) final set: top-50 by largest-box-area, then manually curated to 25 clean images of whole, predominantly red tomatoes to match the webcam deployment scenario.

**Source**: subsampled from Google Open Images (validation + test splits, bbox annotations), class `/m/07j87` (Tomato). Group/depiction boxes excluded; images with 1-6 target boxes, ranked by largest tomato box area, manually reviewed; resized to max 640px; converted to YOLO format; split train 20 / valid 5. Reproducible via `build_dataset.py`. The assignment-suggested Kaggle grocery dataset was verified to contain only tabular product data (no images), so this dataset was created per the assignment's "Find/Create" instruction.

Open Images images are sourced from Flickr under CC BY 2.0; annotations CC BY 4.0 by Google.

- `dataset/` — YOLO layout + `data.yaml` (nc=1)
- `dataset.zip` — same content, for direct upload into the Colab runtime

## Compliance with the revised Lab 1 instructions (instructor email, 2026-09-01)

- **Bounding-box ground truth**: every image carries YOLO-format box labels sourced from Open Images official bbox annotations. ✓
- **Unseen/new classes for the pretrained model**: verified programmatically against `yolo11n.pt` class list — `tomato` is not among the 80 COCO classes (which also motivated dropping `bowl`, a COCO class, from an earlier 3-class draft). ✓
- **YOLO format, directly usable in Colab**: `dataset/` follows the standard YOLO layout (`train/valid` x `images/labels` + `data.yaml`, nc=2); `dataset.zip` uploads straight into the Colab runtime. ✓

Sourcing approach: images collected from the web (Google Open Images) that already carry bounding-box annotations — consistent with the revised instructions, which accept web-collected images and existing annotated datasets; Roboflow Annotate was unnecessary since box labels already existed.
