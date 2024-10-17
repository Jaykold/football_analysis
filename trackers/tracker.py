import os
from ultralytics import YOLO
import supervision as sv
import pickle

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

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            
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

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
            
            print(detection_with_tracks)