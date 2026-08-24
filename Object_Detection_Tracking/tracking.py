import cv2
from ultralytics import YOLO
from collections import Counter

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output_tracking.mp4",
    fourcc,
    fps,
    (width, height)
)

print("Object detection, tracking and recording started.")
print("Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # YOLO detection + tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # Draw boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Get detected object names
    class_names = []

    if results[0].boxes is not None:
        for cls in results[0].boxes.cls:
            class_id = int(cls)
            class_names.append(model.names[class_id])

    # Count objects
    object_counts = Counter(class_names)

    # Display object counts
    y_position = 40

    for object_name, count in object_counts.items():
        text = f"{object_name.capitalize()}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        y_position += 35

    # Total objects
    total_objects = len(class_names)

    cv2.putText(
        annotated_frame,
        f"Total Objects: {total_objects}",
        (20, y_position + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Save frame to video
    out.write(annotated_frame)

    # Show live result
    cv2.imshow(
        "YOLO Detection, Tracking, Counting & Recording",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording saved as: output_tracking.mp4")