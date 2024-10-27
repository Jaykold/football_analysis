import os
import pickle

def load_tracks_from_stub(stub_path):
        if os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        else:
            return None
        
def save_tracks_to_stubs(stub_path, tracks):
    with open(stub_path, 'wb') as f:
        pickle.dump(tracks, f)