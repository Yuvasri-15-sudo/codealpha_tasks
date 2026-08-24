from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Run object detection
results = model("images/test2.jpg", save=True, conf=0.5)

print("Object detection completed successfully!")
print("Detected objects:")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = model.names[class_id]

        print(f"{object_name} - {confidence:.2f}")