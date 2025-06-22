import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv11 model
model = YOLO("best.pt")  # Make sure this is downloaded and placed in the root folder


# Load video
cap = cv2.VideoCapture("15sec_input_720p.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_tracking.mp4", fourcc, 30.0, (1280, 720))

# Deep SORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for result in results.boxes:
        cls_id = int(result.cls)
        conf = float(result.conf)
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = model.names[cls_id]

        if label == "player":
            bbox = [x1, y1, x2 - x1, y2 - y1]  # (x, y, w, h)
            detections.append((bbox, conf, None))

    # Track players
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
