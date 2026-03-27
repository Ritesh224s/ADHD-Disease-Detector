# ADHD Disease Detection - XGBoost Only
# Simplified Training Pipeline

import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Fix UTF-8 encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get absolute path to dataset
base_dir = os.path.dirname(os.path.abspath(__file__))

# Use only the new dataset
file_path = os.path.join(base_dir, "Data", "adhdata.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset 'adhdata.csv' not found")

print(f"✓ Using dataset: adhdata.csv")

print(f"Loading dataset from {file_path}")
df = pd.read_csv(file_path, low_memory=False)

print("\n--- Dataset Info ---")
print(f"Total Samples: {len(df):,}")
print(f"Columns: {df.shape[1]}")

# Handle Missing Values
df.dropna(inplace=True)

# Identify Target Column
target_column = "Class"
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found")

# Encode Target Variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_column].astype(str))
print(f"\nClass Distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} ({label_encoder.classes_[u]}): {c:,} samples ({c/len(y)*100:.1f}%)")

# Select Features
feature_candidates = [col for col in df.columns if col not in {target_column}]
X = df[feature_candidates].select_dtypes(include=[np.number])

print(f"\nSelected {X.shape[1]} numeric features")

# ============================================================================
# 🔧 FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("🔧 FEATURE ENGINEERING")
print("="*70)

X_original = X.copy()
original_features = X.shape[1]

# 1. Statistical Features
print("[OK] Creating statistical features...")
for col in X_original.columns:
    X[f'{col}_mean'] = X_original[col].mean()
    X[f'{col}_std'] = X_original[col].std()

# 2. Global Statistics
print("[OK] Creating global statistical features...")
X['all_channels_mean'] = X_original.mean(axis=1)
X['all_channels_std'] = X_original.std(axis=1)
X['all_channels_min'] = X_original.min(axis=1)
X['all_channels_max'] = X_original.max(axis=1)
X['all_channels_range'] = X_original.max(axis=1) - X_original.min(axis=1)

# 3. Top Channel Interactions
print("[OK] Creating interaction features...")
top_channels = X_original.std().nlargest(3).index.tolist()
for i in range(len(top_channels)):
    for j in range(i+1, len(top_channels)):
        X[f'{top_channels[i]}_x_{top_channels[j]}'] = X_original[top_channels[i]] * X_original[top_channels[j]]

print(f"Original Features: {original_features}")
print(f"New Features Created: {X.shape[1] - original_features}")
print(f"Total Features: {X.shape[1]}")

# Handle NaN/Inf values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Normalize Data
print("\n[OK] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 📊 TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("📊 DATA SPLIT")
print("="*70)

# Use 80% of all data
sample_size = int(len(X_scaled) * 0.8)
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_scaled_sample = X_scaled[sample_indices]
y_sample = y[sample_indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
print(f"Training samples: {len(X_train):,} | Test samples: {len(X_test):,}")

# ============================================================================
# 🚀 TRAINING XGBOOST
# ============================================================================
print("\n" + "="*70)
print("🚀 TRAINING XGBOOST")
print("="*70)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
scale_pos_weight = class_weights[1] / class_weights[0]

# Classes are already 50/50 balanced - SMOTE not needed
# Proceeding directly with training data

# Grid search for better parameters (Reduced - Option A)
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

clf_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    n_jobs=-1,
    eval_metric='logloss',
    random_state=42,
    verbosity=1
)

print("[OK] Running GridSearchCV with CPU processing...")
grid_search = GridSearchCV(
    clf_xgb, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=3  # 👈 Change this to 3 for detailed output
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

clf_xgb = grid_search.best_estimator_
print("[OK] GridSearchCV completed - Using best model")

y_pred = clf_xgb.predict(X_test)
y_pred_prob = clf_xgb.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✓ XGBoost Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✓ AUC-ROC Score: {auc_score:.4f}")

# ============================================================================
# 💾 SAVE MODELS
# ============================================================================
print("\n" + "="*70)
print("💾 SAVING MODELS")
print("="*70)

# Save XGBoost model
xgb_model_file = os.path.join(base_dir, "xgboost_adhd_model.pkl")
with open(xgb_model_file, 'wb') as f:
    pickle.dump(clf_xgb, f)
print(f"[OK] XGBoost model saved: xgboost_adhd_model.pkl")

# Save scaler
scaler_file = os.path.join(base_dir, "xgboost_scaler.pkl")
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[OK] Scaler saved: xgboost_scaler.pkl")

# Save model info
info_file = os.path.join(base_dir, "model_info.pkl")
with open(info_file, 'wb') as f:
    pickle.dump({
        'feature_names': X.columns.tolist(),
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'num_features': X.shape[1]
    }, f)
print(f"[OK] Model info saved: model_info.pkl")

# ============================================================================
# 📊 SAVE METRICS
# ============================================================================
metrics_file = os.path.join(base_dir, "model_metrics.txt")
with open(metrics_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ADHD DISEASE DETECTION - XGBOOST MODEL\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Training Samples: {len(X_train):,} (80%)\n")
    f.write(f"Testing Samples: {len(X_test):,} (20%)\n")
    f.write(f"Original Features: {original_features}\n")
    f.write(f"Engineered Features: {X.shape[1] - original_features}\n")
    f.write(f"Total Features: {X.shape[1]}\n")
    f.write(f"Classes: {len(np.unique(y))} (ADHD vs Control)\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"AUC-ROC Score: {auc_score:.4f}\n\n")
    
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*70 + "\n")
    f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    f.write("\nCONFUSION MATRIX\n")
    f.write("-"*70 + "\n")
    f.write(f"True Negatives: {cm[0,0]:,}\n")
    f.write(f"False Positives: {cm[0,1]:,}\n")
    f.write(f"False Negatives: {cm[1,0]:,}\n")
    f.write(f"True Positives: {cm[1,1]:,}\n\n")
    
    f.write("SENSITIVITY & SPECIFICITY\n")
    f.write("-"*70 + "\n")
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
    specificity = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0
    f.write(f"Sensitivity (True Positive Rate): {sensitivity:.4f}\n")
    f.write(f"Specificity (True Negative Rate): {specificity:.4f}\n")

print(f"[OK] Metrics saved: model_metrics.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
print(f"\nMODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   AUC-ROC: {auc_score:.4f}")
print(f"\nFEATURES:")
print(f"   Total Features: {X.shape[1]} ({original_features} original + {X.shape[1] - original_features} engineered)")
print(f"\nFILES SAVED:")
print(f"   ✓ xgboost_adhd_model.pkl")
print(f"   ✓ xgboost_scaler.pkl")
print(f"   ✓ model_info.pkl")
print(f"   ✓ model_metrics.txt")
print("="*70 + "\n")
