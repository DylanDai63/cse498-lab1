# Step 1 — Webcam object detection

## Run

```
conda activate cse498   # any env with ultralytics + opencv-python
cd webcam-object-recognition
python main.py            # pretrained COCO yolo11n (green boxes)
python main_finetuned.py  # our fine-tuned tomato model (red boxes)
```

Press `q` in the video window to quit.

- `main.py` loads `models/yolo11n.pt` (committed). Only change vs upstream: `verbose=False` on predict.
- `main_finetuned.py` loads `../../step2-finetune/results/weights/best.pt` (committed) with conf=0.25, to redo the experiment after fine-tuning.
- `results-pretrained/` and `results-finetuned/` hold the captured evidence; file names encode label and confidence (`ok__`/`miss__`/`finetuned__`).
