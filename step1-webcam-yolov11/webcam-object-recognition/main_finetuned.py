import cv2
from ultralytics import YOLO

# Same webcam loop as main.py, but loads OUR fine-tuned weights (tomato only)
# instead of the COCO-pretrained model — used to redo the Step 1 experiment
# after fine-tuning, as required by the assignment.
model = YOLO("../../step2-finetune/results/weights/best.pt")

cap = cv2.VideoCapture(0)  # Open the default webcam

frame_number = -1
ret = True

while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret:
        # conf=0.25: standard visualization threshold (model is confident enough now)
        detections = model.predict(frame, conf=0.25, verbose=False)
        for detection in detections:
            boxes = detection.boxes.xyxy
            scores = detection.boxes.conf
            classes = detection.boxes.cls
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
        cv2.imshow("YOLOv11 Fine-tuned (tomato)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
