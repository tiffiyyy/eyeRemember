import numpy as np
from face_reid import preprocess, getH, getW
import pickle
from utils import save_face_db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(embedding, face_db, threshold):
    best_score = -1
    best_match = "Unknown"
    for name, db_embedding in face_db.items():
        score = cosine_similarity(embedding, db_embedding)
        if score > best_score and score > threshold:
            best_score = score
            best_match = name
    return best_match

# for every image processed, generate an image ID for each person 
    #extract name from frontend name upload, should linked with the previously extracted image

def save_face_db(image,name):
    embeddings = preprocess(image, threshold=0.3, width=getW(), height=getH())
    name = name
    # desc= desc


    try:
        with open("data/face_db.pkl", "rb") as f:
            profiles = pickle.load(f)
    except FileNotFoundError:
        profiles = {}

    if embeddings:
        profiles[name] = embeddings[0]
    
    # populates a pickle file with all profiles 
    with open("face_db.pkl", "wb") as f:
        pickle.dump(profiles, f)