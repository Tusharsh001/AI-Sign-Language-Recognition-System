# src/preprocessing.py

import numpy as np

def extract_landmark_vector(hand_landmarks):
    """
    Extract 63-dimensional landmark vector
    (21 landmarks × x, y, z)
    """
    landmark_vector = []
    for landmark in hand_landmarks.landmark:
        landmark_vector.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmark_vector)
