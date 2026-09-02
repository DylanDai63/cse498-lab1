# Step 2 — Fine-tuning dataset

50 images / 1 class: `tomato` — the Step 1 failure case chosen as the fine-tuning target (the pretrained model detected tomatoes as apples). An earlier 2-class draft (tomato+toilet_paper) showed the toilet_paper class failed to learn with only ~12 training images (mAP50 0.01 vs tomato 0.59) — an instructive data-quantity lesson noted in the report; the final experiment focuses on tomato with the full 50-image budget.

**Source**: subsampled from Google Open Images (validation + test splits, bbox annotations), class `/m/07j87` (Tomato). Group/depiction boxes excluded; images with 1-6 target boxes preferred; resized to max 640px; converted to YOLO format; split train 42 / valid 8 (85/15); 137 training boxes. Reproducible via `build_dataset.py`. The assignment-suggested Kaggle grocery dataset was verified to contain only tabular product data (no images), so this dataset was created per the assignment's "Find/Create" instruction.

Open Images images are sourced from Flickr under CC BY 2.0; annotations CC BY 4.0 by Google.

- `dataset/` — YOLO layout + `data.yaml` (nc=1)
- `dataset.zip` — same content, for direct upload into the Colab runtime

## Compliance with the revised Lab 1 instructions (instructor email, 2026-09-01)

- **Bounding-box ground truth**: every image carries YOLO-format box labels sourced from Open Images official bbox annotations. ✓
- **Unseen/new classes for the pretrained model**: verified programmatically against `yolo11n.pt` class list — `tomato` is not among the 80 COCO classes (which also motivated dropping `bowl`, a COCO class, from an earlier 3-class draft). ✓
- **YOLO format, directly usable in Colab**: `dataset/` follows the standard YOLO layout (`train/valid` x `images/labels` + `data.yaml`, nc=2); `dataset.zip` uploads straight into the Colab runtime. ✓

Sourcing approach: images collected from the web (Google Open Images) that already carry bounding-box annotations — consistent with the revised instructions, which accept web-collected images and existing annotated datasets; Roboflow Annotate was unnecessary since box labels already existed.
