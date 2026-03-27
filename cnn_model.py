# ADHD Disease Detection - XGBoost CPU Training Pipeline
import os
import sys
import numpy as np
import pandas as pd

# Fix UTF-8 encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "Data", "adhdata.csv")

print("Loading ADHD Dataset for XGBoost CPU Training...")
print(f"Path: {data_path}")

# Load your dataset
data = pd.read_csv(data_path, low_memory=False)

# Separate features and target
target_column = "Class"
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found")

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data[target_column].astype(str))

# Select numeric features
feature_candidates = [col for col in data.columns if col not in {target_column}]
X = data[feature_candidates].select_dtypes(include=[np.number])

print(f"\n✓ Dataset Shape: {X.shape}")
print(f"✓ Target Classes: {label_encoder.classes_}")
print(f"✓ Class Distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  - {label_encoder.classes_[u]}: {c:,} samples ({c/len(y)*100:.1f}%)")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training samples: {len(X_train):,}")
print(f"✓ Testing samples: {len(X_test):,}")

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and Train XGBoost Model with CPU
print("\n" + "="*70)
print("🚀 BUILDING AND TRAINING XGBOOST MODEL (CPU)")
print("="*70)

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='auto',
    device='cpu',
    random_state=42,
    verbosity=1,
    eval_metric='logloss'
)

print("\n📊 Training XGBoost with CPU...")
model.fit(X_train_scaled, y_train)
print("✓ Training completed!")

# Evaluate on test set
print("\n" + "="*70)
print("📈 EVALUATING XGBOOST MODEL")
print("="*70)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✓ Accuracy:  {accuracy*100:.2f}%")
print(f"✓ Precision: {precision*100:.2f}%")
print(f"✓ Recall:    {recall*100:.2f}%")
print(f"✓ F1-Score:  {f1*100:.2f}%")
print(f"✓ ROC-AUC:   {roc_auc*100:.2f}%")

# Classification Report
print("\n" + "-"*70)
print("Classification Report:")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Plot Feature Importance
print("\n📊 Generating feature importance plot...")
plt.figure(figsize=(12, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]
plt.barh(range(len(indices)), importances[indices])
plt.xlabel('Importance', fontsize=12)
plt.title('XGBoost Feature Importance (Top 20)', fontsize=14, fontweight='bold')
plt.tight_layout()
feature_importance_path = os.path.join(base_dir, 'xgboost_feature_importance.png')
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: xgboost_feature_importance.png")

# Plot Confusion Matrix
print("📊 Generating confusion matrix plot...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cbar_kws={'label': 'Count'})
plt.title('XGBoost Model - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
xgb_cm_path = os.path.join(base_dir, 'xgboost_confusion_matrix.png')
plt.savefig(xgb_cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: xgboost_confusion_matrix.png")

# Save the trained model
print("\n💾 Saving model and scaler...")
xgb_model_path = os.path.join(base_dir, 'xgboost_adhd_model.pkl')
xgb_scaler_path = os.path.join(base_dir, 'xgboost_scaler.pkl')

pickle.dump(model, open(xgb_model_path, 'wb'))
pickle.dump(scaler, open(xgb_scaler_path, 'wb'))

print(f"✅ XGBoost model saved: {xgb_model_path}")
print(f"✅ Scaler saved: {xgb_scaler_path}")

# Save metrics to file
metrics_path = os.path.join(base_dir, 'xgboost_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("XGBoost ADHD Detection Model - Performance Metrics (CPU)\n")
    f.write("="*50 + "\n\n")
    f.write(f"Accuracy:  {accuracy*100:.2f}%\n")
    f.write(f"Precision: {precision*100:.2f}%\n")
    f.write(f"Recall:    {recall*100:.2f}%\n")
    f.write(f"F1-Score:  {f1*100:.2f}%\n")
    f.write(f"ROC-AUC:   {roc_auc*100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print(f"✅ Metrics saved: {metrics_path}")

print("\n" + "="*70)
print("✅ XGBOOST CPU TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)