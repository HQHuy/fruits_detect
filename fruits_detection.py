import cv2
from ultralytics import YOLO
import time

# Cấu hình
MODEL_PATH = "/Users/hqhyy/Downloads/new/Model/runs/train_fruits_final/weights/best.pt"
WEBCAM_INDEX = 0
CONF_THRESHOLD = 0.7

# Tải model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit(1)

# Mở webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit(1)

print("Bắt đầu phát hiện với YOLOv8. Nhấn 'q' để thoát.")
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    results = model(frame)

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tính và hiển thị FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Đếm và hiển thị số đối tượng
    num_objects = len([box for box in result.boxes if box.conf[0].item() > CONF_THRESHOLD])
    cv2.putText(frame, f"Objects: {num_objects}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('detection.jpg', frame)
        print("Đã lưu detection.jpg")

cap.release()
cv2.destroyAllWindows()
print("Đã dừng phát hiện.")