# ADHD project
# Load Dataset
import os
import sys
import pandas as pd
import numpy as np

# Fix UTF-8 encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CPU mode enabled for compatibility
print("=" * 70)
print("🔧 DEVICE CONFIGURATION")
print("=" * 70)
print("CPU mode enabled for model training")

# Set matplotlib to use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')

# Get absolute path to dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Dataset", "adhdata.csv")

# Check if dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

print(f"Loading dataset from {file_path}")
df = pd.read_csv(file_path, low_memory=False)

# 2️⃣ **Check Dataset Info**
print("\n--- Dataset Preview ---")
print(df.head())

# 3️⃣ **Handle Missing Values**
df.dropna(inplace=True)

# 4️⃣ **Identify Target Column**
target_column = "Class"
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found")

# 5️⃣ **Encode Target Variable**
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_column].astype(str))
print(f"\nTarget class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# 6️⃣ **Select Features (Exclude ID and Target)**
feature_candidates = [col for col in df.columns if col not in {target_column, "ID"}]
X = df[feature_candidates].select_dtypes(include=[np.number])

if X.empty:
    raise ValueError("No numeric features remain after filtering columns")

print(f"\nSelected {X.shape[1]} numeric features for training")

# 7️⃣ **Normalize Data**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️⃣ **Train-test split**
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- 1D-CNN PREPARATION ---
# Reshape input for 1D-CNN: [samples, time_steps/features, channels]
# Here, we treat features as time_steps since it's tabular data
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Train Shape (CNN): {X_train_cnn.shape}, Test Shape (CNN): {X_test_cnn.shape}")

# 9️⃣ **Build 1D-CNN Model Architecture**
print("\n--- Building 1D-CNN Model ---")

model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    
    # 1st Conv Layer
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # 2nd Conv Layer
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularization to prevent overfitting
    
    # Output Layer (Binary Classification: 0 or 1)
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks (Save best model & Stop if not improving)
checkpoint_path = os.path.join(base_dir, "adhd_cnn_model.keras")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
]

# Train the model
print("\n--- Training 1D-CNN ---")
history = model.fit(
    X_train_cnn, y_train,
    epochs=10, # Adjustable
    batch_size=64,
    validation_data=(X_test_cnn, y_test),
    callbacks=callbacks,
    verbose=1
)
print("[OK] Model training completed")

# Save the scaler (needed for new data prediction)
scaler_file = os.path.join(base_dir, "scaler.pkl")
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[OK] Scaler saved to 'scaler.pkl'")

# 1️⃣0️⃣ **Make Predictions**
# Predict probabilities
y_pred_prob = model.predict(X_test_cnn)
# Convert probabilities to binary classes (Threshold 0.5)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 1️⃣1️⃣ **Evaluate Model**
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"🔹 Model Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Save Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "training_history.png"))
print("[OK] Training history plot saved")

# 1️⃣2️⃣ **Confusion Matrix**
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (CNN)")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
print("[OK] Confusion matrix saved")
plt.close() # Close plot to free memory

print("\n=== TRAINING COMPLETE ===")
print(f"All files saved to: {base_dir}")