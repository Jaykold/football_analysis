from ultralytics import YOLO

video_path = "input_videos/08fd33_4.mp4"

model = YOLO('models/best.pt')  # Load model

results = model.predict(video_path, save=True) # Inference on video stream

for box in results[0].boxes:
    print(box)  # print class and box coordinates