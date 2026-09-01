# Step 2 — Fine-tuning dataset

60 images / 3 classes: `tomato` (22), `toilet_paper` (16), `bowl` (22) — the object categories the pretrained YOLOv11n misclassified in Step 1 (tomato->apple, toilet paper->cup/toilet, bowl->sink).

**Source**: subsampled from Google Open Images (validation + test splits, bbox annotations), classes `/m/07j87`, `/m/09gtd`, `/m/04kkgm`. Group/depiction boxes excluded; images with 1-4 target boxes preferred; resized to max 640px; converted to YOLO format. Reproducible via `build_dataset.py`. The assignment-suggested Kaggle grocery dataset was verified to contain only tabular product data (no images), so this dataset was created per the assignment's "Find/Create" instruction.

Open Images images are sourced from Flickr under CC BY 2.0; annotations CC BY 4.0 by Google.

- `dataset/` — train (51) / valid (9), YOLO layout + `data.yaml`
- `dataset.zip` — same content, for direct upload into the Colab runtime
