# src/collector.py

import cv2
import mediapipe as mp
import pandas as pd
import os
import time

from config import CLASSES, TARGET_SAMPLES, DATASET_PATH, SAVE_INTERVAL
from preprocessing import extract_landmark_vector


class LandmarkDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.classes = CLASSES
        self.current_class = self.classes[0]
        self.samples_collected = 0
        self.target_samples = TARGET_SAMPLES

        self.create_dataset_file()

    def create_dataset_file(self):
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(DATASET_PATH):
            header = []
            for i in range(21):
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            header.append('class_label')

            pd.DataFrame(columns=header).to_csv(DATASET_PATH, index=False)

    def save_landmark_data(self, landmark_vector):
        row = list(landmark_vector) + [self.current_class]

        header = []
        for i in range(21):
            header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        header.append('class_label')

        pd.DataFrame([row], columns=header).to_csv(
            DATASET_PATH, mode='a', header=False, index=False
        )

        self.samples_collected += 1

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def move_to_next_class(self):
        idx = self.classes.index(self.current_class)
        if idx + 1 < len(self.classes):
            self.current_class = self.classes[idx + 1]
            self.samples_collected = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        auto_collect = False
        last_save = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.draw_landmarks(frame, hand_landmarks)

                if auto_collect:
                    now = time.time()
                    if now - last_save >= SAVE_INTERVAL:
                        vector = extract_landmark_vector(hand_landmarks)
                        self.save_landmark_data(vector)
                        last_save = now

                        if self.samples_collected >= self.target_samples:
                            self.move_to_next_class()
                            auto_collect = False

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                auto_collect = not auto_collect
            elif key == ord('n'):
                self.move_to_next_class()

        cap.release()
        cv2.destroyAllWindows()
