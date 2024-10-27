import os
import pickle
from ultralytics import YOLO
import supervision as sv
import cv2
from utils import get_center_of_bbox, get_bbox_width, load_tracks_from_stub, save_tracks_to_stubs

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            detections_batch = self.model.predict(batch, conf=0.1)
            detections.extend(detections_batch)
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None:
            tracks = load_tracks_from_stub(stub_path)
            if tracks is not None:
                return tracks
             
        detections = self.detect_frames(frames)

        tracks = {
            "players": [], # {0:{"box": [0,0,0,0]}, 1:{"box": [0,1,0,2]}, 5:{"box": [0,0,0,0]}}
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # convert Goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Store tracks
            for track in detection_with_tracks:
                bbox = track[0].tolist()
                cls_id = track[3]
                track_id = track[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for track in detection_supervision:
                bbox = track[0].tolist()
                cls_id = track[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path:
            save_tracks_to_stubs(stub_path, tracks)
            
        return tracks
        
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        center_x, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        center = (center_x, y2)
        axes = (int(width), int(0.35*width))
        angle = 0.0
        start_angle = -65
        end_angle = 255
        colour = color
        thickness = 2
        line_type = cv2.LINE_4

        cv2.ellipse(frame, center, axes, angle, start_angle, end_angle, colour, thickness, line_type)
        #cv2.putText(frame, str(track_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
        
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            if frame_num < len(tracks['players']) and frame_num < len(tracks['referees']) and frame_num < len(tracks['ball']):
                #print(f"Drawing annotations for frame_num {frame_num}, {tracks["players"][frame_num]}")
                player_dicts: dict = tracks["players"][frame_num]
                referee_dicts: dict = tracks["referees"][frame_num]
                ball_dicts: dict = tracks["ball"][frame_num]

                # Draw players
                for track_id, player_data in player_dicts.items():
                    player_bbox = player_data["bbox"]
                    frame = self.draw_ellipse(frame, player_bbox, (173,255,47), track_id)

                for track_id, referee_data in referee_dicts.items():
                    referee_bbox = referee_data["bbox"]
                    frame = self.draw_ellipse(frame, referee_bbox, (255, 165, 0), track_id)

                output_video_frames.append(frame)
            else:
                print(f"Skipping frame {frame_num} due to missing data")
            
        return output_video_frames

