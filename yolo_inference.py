from ultralytics import YOLO

model = YOLO('yolov8')  # Load model

results = model.predict(save=True, stream=True) # Inference on video stream

for box in results[0].boxes:
    print(box)  # print class and box coordinates