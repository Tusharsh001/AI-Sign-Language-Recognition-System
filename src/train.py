import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

def train_landmark_model():
    print("🚀 Starting Landmark Model Training...")
    print("=" * 50)
    
    # 1. Load your clean dataset
    df = pd.read_csv('asl_landmark_dataset.csv')
    print(f"📊 Dataset loaded: {len(df)} samples, {df.shape[1]-1} features")
    
    # 2. Prepare features and labels
    X = df.iloc[:, :-1].values  # All landmark coordinates (63 features)
    y = df.iloc[:, -1].values   # Class labels
    
    # 3. Encode labels (A->0, B->1, C->2, etc.)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"🎯 Classes: {list(label_encoder.classes_)}")
    print(f"🔢 Encoded: {len(label_encoder.classes_)} total classes")
    
    # 4. Split data (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded  # Maintain class distribution
    )
    
    print(f"📁 Training set: {X_train.shape[0]} samples")
    print(f"📁 Validation set: {X_val.shape[0]} samples")
    
    # 5. Build the model
    model = keras.Sequential([
        # Input layer
        keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Hidden layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        # Output layer
        keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    # 6. Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model architecture:")
    model.summary()
    
    # 7. Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_accuracy',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 8. Train the model
    print("\n🎯 Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Save the model and metadata
    model.save('asl_landmark_model.keras')
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    print("💾 Model saved: asl_landmark_model.keras")
    print("💾 Class names saved: label_encoder_classes.npy")
    
    # 10. Evaluate the model
    print("\n📊 Model Evaluation:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"🎯 Validation Accuracy: {val_accuracy:.2%}")
    print(f"📉 Validation Loss: {val_loss:.4f}")
    
    # 11. Generate predictions for detailed analysis
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_val, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # 12. Plot training history (no seaborn needed)
    plot_training_history(history)
    
    # 13. Simple confusion matrix (no seaborn needed)
    plot_simple_confusion_matrix(y_val, y_pred_classes, label_encoder.classes_)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_simple_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix without seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Create a simple heatmap using matplotlib
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"✅ GPU detected: {physical_devices[0]}")
    else:
        print("ℹ️  Training on CPU")
    
    # Train the model
    model, history = train_landmark_model()
    
    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("Next steps:")
    print("1. Check training_history.png and confusion_matrix.png")
    print("2. Run real-time test with your webcam")
    print("3. Use the model for ASL recognition!")