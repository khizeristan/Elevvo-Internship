# gtsrb_train.py
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "D:/Eleevo/Task4- Signal recognition/dataset"
train_csv = os.path.join(dataset_dir, "Train.csv")
test_csv = os.path.join(dataset_dir, "Test.csv")
train_dir = os.path.join(dataset_dir, "Train")
test_dir = os.path.join(dataset_dir, "Test")

# Function to load data
def load_data(csv_file, data_dir, img_size=(32, 32)):
    data = pd.read_csv(csv_file)
    images, labels = [], []
    for _, row in data.iterrows():
        # Normalize path separators
        relative_path = row['Path'].replace("\\", "/")
        img_path = os.path.join(data_dir, relative_path)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(row['ClassId'])
        else:
            print(f"[WARNING] File not found: {img_path}")

    return np.array(images), np.array(labels)

print("✅ Loading training data...")
X_train, y_train = load_data(train_csv, dataset_dir)
print("✅ Loading testing data...")
X_test, y_test = load_data(test_csv, dataset_dir)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Train/validation split
X_train, X_val, y_train_cat, y_val_cat = train_test_split(
    X_train, y_train_cat, test_size=0.2, random_state=42, stratify=y_train
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("🚀 Training started...")
history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val_cat)
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Test accuracy: {test_acc:.4f}")

# Save model
model.save("traffic_sign_model.h5")
print("💾 Model saved as traffic_sign_model.h5")
