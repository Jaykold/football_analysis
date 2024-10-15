from utils import read_video, save_video

def main():
    video_path = 'input_videos/08fd33_4.mp4'

    # Read video
    video_frames = read_video(video_path)

    # Save video
    save_video(video_frames, video_path)

if __name__ == '__main__':
    main()