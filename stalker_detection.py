import os
import sys
import time
import warnings
import cv2
import torch
import geocoder
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client

# -----------------------
# Suppress warnings
# -----------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# Twilio Config
# -----------------------
ACCOUNT_SID = "YOUR ACCT SID"
AUTH_TOKEN = "YOUR AUTH TOKEN"
FROM_NUMBER = " "  # Twilio number
TO_NUMBER = " "   # Your personal number

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_alert():
    """Send SMS alert with location"""
    g = geocoder.ip('me')
    if g.latlng:
        location_url = f"https://www.google.com/maps?q={g.latlng[0]},{g.latlng[1]}"
    else:
        location_url = "Location not available"

    message = client.messages.create(
        body=f"🚨 ALERT: Person followed you for over 1 minute! Location: {location_url}",
        from_=FROM_NUMBER,
        to=TO_NUMBER
    )
    print(f"[ALERT SENT] SID: {message.sid}")

# -----------------------
# Load YOLOv5 Model
# -----------------------
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # Confidence threshold
model.classes = [0]  # Only detect 'person'

# -----------------------
# Initialize DeepSORT
# -----------------------
tracker = DeepSort(max_age=30)

# Track time each ID has been seen
seen_times = {}
alert_sent_ids = set()

# -----------------------
# Start Webcam
# -----------------------
cap = cv2.VideoCapture(0)  # Use webcam index 0
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    sys.exit()

print("[INFO] Starting video stream... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Run YOLOv5
    results = model(frame)
    detections = results.xyxy[0]

    # Convert YOLO detections → DeepSORT format
    person_dets = []
    for *box, conf, cls in detections.tolist():
        if int(cls) == 0:  # Only person class
            x1, y1, x2, y2 = box
            person_dets.append(([x1, y1, x2, y2], float(conf), "person"))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(person_dets, frame=frame)

    current_time = time.time()

    # Process each tracked person
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw tracking box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Track how long each person stays
        if track_id not in seen_times:
            seen_times[track_id] = current_time
        else:
            elapsed = current_time - seen_times[track_id]
            if elapsed > 60 and track_id not in alert_sent_ids:
                print(f"[WARNING] Person ID {track_id} stayed > 1 min!")
                send_alert()
                alert_sent_ids.add(track_id)

    # Show live feed
    cv2.imshow("YOLOv5 + DeepSORT Stalker Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
