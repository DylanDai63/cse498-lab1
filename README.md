# CSE 398/498 · Lab 1 — Object Detection with YOLOv11 & YOLOE

Student: Hengde Dai (hed424)

This repository contains all three experiments and the report for Lab 1.

## Environment

- Python 3.10 (conda env), PyTorch 2.5.1 + CUDA 12.1
- `ultralytics` 8.4.135, `opencv-python`
- Tested on Windows 11, NVIDIA RTX 3070 Ti (8 GB)

## Contents

| Folder | Experiment | Status |
|---|---|---|
| `step1-webcam-yolov11/` | Pretrained YOLOv11 on live webcam (grocery/household objects) | in progress |
| `step2-finetune/` | Fine-tuning YOLOv11 on a ~50-image custom dataset; before/after comparison | pending |
| `step3-yoloe-openset/` | Open-set (zero-shot) object detection with YOLOE | pending |
| `report/` | 6-page report | pending |

Run instructions for each experiment are in the corresponding folder's README.

## Attribution

- `step1-webcam-yolov11/webcam-object-recognition/` is vendored from
  [valeriouberti/webcam-object-recognition](https://github.com/valeriouberti/webcam-object-recognition)
  @ `e1279fa`, as instructed by the assignment.
- Steps 2 and 3 follow the official Roboflow notebooks linked in the assignment.
- AI assistance (Claude) was used for environment setup, debugging support, and editing; cited per course policy in the report.
