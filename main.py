from utils import read_video, save_video
from trackers import Tracker

def main():
    video_input_path = 'input_videos/08fd33_4.mp4'
    video_output_path = 'output_videos/output_video.avi'
    model_path = 'models/best.pt'
    stub_path = 'stubs/track_stubs.pkl'

    tracker = Tracker(model_path)

    # Read video
    video_frames = read_video(video_input_path)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=stub_path)
    
    # Draw Output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save video
    save_video(output_video_frames, video_output_path)

if __name__ == '__main__':
    main()