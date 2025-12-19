# quick_webcam_test.py
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

def quick_webcam_test():
    # Load your amazing model
    model = tf.keras.models.load_model('asl_landmark_model.keras')
    class_names = np.load('label_encoder_classes.npy', allow_pickle=True)
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    
    print("🎥 REAL-WORLD TEST")
    print("Show ASL letters to your webcam")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Predict
            prediction = model.predict(np.array([landmarks]), verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            predicted_letter = class_names[predicted_class]
            
            # Draw result
            color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
            cv2.putText(frame, f"Prediction: {predicted_letter}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
            )
        
        cv2.imshow('ASL Recognition - REAL TEST', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    quick_webcam_test()