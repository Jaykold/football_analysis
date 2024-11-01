import cv2

video_dir = "runs/detect/predict2/08fd33_4.avi"

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:  
        print("Error: output_video_frames is empty. No video to save.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 24
    width = output_video_frames[0].shape[1]
    height = output_video_frames[0].shape[0]
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        video.write(frame)
    video.release()
        