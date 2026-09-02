# Step 3 — Open-set object detection with YOLOE

Followed the assigned Roboflow Colab (`zero_shot_object_detection_and_segmentation_with_yoloe.ipynb`, executed copy with outputs included).

- **Text prompt**: class names (e.g. `["dog", "eye", "nose", "tongue"]`) are encoded once by a CLIP-style text encoder; the embeddings act as classification-head weights, so YOLOE detects categories it was never explicitly trained on — see `results/text-prompt__dog-eye-nose-tongue.png`.
- **Visual prompt**: a reference bounding box serves as the prompt instead of text (YOLOEVPSegPredictor section).
- Video-processing sections repeat the same concepts on video and were skipped.

Weights: `yoloe-v8l-seg.pt` from the official jameslahm/yoloe HuggingFace repo (pulled in-notebook, not committed).
