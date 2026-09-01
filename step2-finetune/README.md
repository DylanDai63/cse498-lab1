# Step 2 — Fine-tuning dataset

50 images / 2 classes: `tomato` (34), `toilet_paper` (16) — the two failure cases from Step 1 chosen as fine-tuning targets (tomato was detected as apple, toilet paper as cup/toilet).

**Source**: subsampled from Google Open Images (validation + test splits, bbox annotations), classes `/m/07j87` (Tomato) and `/m/09gtd` (Toilet paper). Group/depiction boxes excluded; images with 1-4 target boxes preferred; resized to max 640px; converted to YOLO format; split train 42 / valid 8 (85/15). Reproducible via `build_dataset.py`. The assignment-suggested Kaggle grocery dataset was verified to contain only tabular product data (no images), so this dataset was created per the assignment's "Find/Create" instruction.

Open Images images are sourced from Flickr under CC BY 2.0; annotations CC BY 4.0 by Google.

- `dataset/` — YOLO layout + `data.yaml` (nc=2)
- `dataset.zip` — same content, for direct upload into the Colab runtime
