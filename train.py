import os
import json
import numpy as np
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from datetime import datetime

class HandGestureTrainer:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.X = []
        self.y = []
        self.labels = {}
        self.model = None
        self.model_metadata = {}

    def extract_hand_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract and normalize coordinates relative to wrist
        landmarks = results.multi_hand_landmarks[0]
        wrist = landmarks.landmark[0]
        points = []
        
        for landmark in landmarks.landmark:
            # Normalize coordinates relative to wrist position
            points.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        return points

    def load_dataset(self):
        print("Loading and processing dataset...")
        for idx, gesture_folder in enumerate(os.listdir(self.dataset_path)):
            self.labels[idx] = gesture_folder
            folder_path = os.path.join(self.dataset_path, gesture_folder)
            
            if not os.path.isdir(folder_path):
                continue
                
            print(f"Processing {gesture_folder}...")
            for image_file in os.listdir(folder_path):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(folder_path, image_file)
                features = self.extract_hand_features(image_path)
                
                if features is not None and len(features) == 63:  # 21 landmarks × 3 coordinates
                    self.X.append(features)
                    self.y.append(idx)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print(f"Processed {len(self.X)} valid images")
        return len(self.labels)

    def build_model(self, num_classes):
        # Input shape: 21 landmarks × 3 coordinates = 63 features
        input_dim = 63
        
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        if len(self.X) == 0:
            raise ValueError("No data loaded! Run load_dataset first.")

        num_classes = len(self.labels)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=validation_split, random_state=42
        )

        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        self.build_model(num_classes)
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test)
        )
        
        # Save model metadata
        self.model_metadata = {
            "model_name": "hand_gesture_recognition",
            "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "number_of_classes": num_classes,
            "labels": self.labels,
            "input_shape": (63,),
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "validation_split": validation_split
            },
            "training_results": {
                "final_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                "final_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1])
            },
            "preprocessing": {
                "mediapipe_confidence": 0.5,
                "num_landmarks": 21,
                "coordinates_per_landmark": 3
            }
        }
        
        return history

    def save_model(self, model_dir="model_output"):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save the model in H5 format
        model_path = os.path.join(model_dir, "hand_gesture_model.h5")
        self.model.save(model_path, save_format='h5')
        
        # Save the metadata
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=4)
            
        print(f"Model saved as {model_path}")
        print(f"Metadata saved as {metadata_path}")

def main():
    trainer = HandGestureTrainer()
    num_classes = trainer.load_dataset()
    
    if num_classes > 0:
        trainer.train()
        trainer.save_model()
    else:
        print("No valid classes found in the dataset!")

if __name__ == "__main__":
    main()
