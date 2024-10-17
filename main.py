from utils import read_video, save_video
from trackers import Tracker

def main():
    video_input_path = 'input_videos/08fd33_4.mp4'
    video_output_path = 'output_videos/output_video.avi'
    model_path = 'models/best.pt'

    # Read video
    video_frames = read_video(video_input_path)

    tracker = Tracker(model_path)

    tracker.get_object_tracks(video_frames)

    # Save video
    save_video(video_frames, video_output_path)

if __name__ == '__main__':
    main()