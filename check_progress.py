import os

print("\n" + "="*70)
print("ENSEMBLE MODEL TRAINING PROGRESS")
print("="*70 + "\n")

# Check files
files_to_check = {
    'confusion_matrix.png': 'Confusion Matrix',
    'feature_importance.png': 'Feature Importance Chart',
    'model_comparison.png': 'Model Comparison Chart',
    'roc_curve.png': 'ROC Curve',
    'prediction_analysis.png': 'Prediction Analysis',
    'model_metrics.txt': 'Detailed Metrics Report',
    'model_xgboost.pkl': 'XGBoost Trained Model',
    'model_gradboost.pkl': 'Gradient Boosting Model',
    'model_randomforest.pkl': 'Random Forest Model',
    'scaler.pkl': 'Feature Scaler'
}

print("Status of Generated Files:\n")
for file, description in files_to_check.items():
    exists = os.path.exists(file)
    status = "✓ DONE" if exists else "⏳ IN PROGRESS"
    print(f"  {status:12} {description:40} ({file})")

print("\n" + "="*70)
print("When complete, all files should show ✓ DONE")
print("="*70 + "\n")

# Check for errors
if os.path.exists('model_xgboost.pkl'):
    print("✅ XGBoost model trained successfully!")
    
if os.path.exists('model_metrics.txt'):
    with open('model_metrics.txt', 'r') as f:
        content = f.read()
        if 'ENSEMBLE' in content and 'BEST' in content:
            print("✅ Ensemble model metrics available!")
            # Extract accuracy
            for line in content.split('\n'):
                if 'Ensemble Accuracy' in line or 'Enhanced Ensemble' in line:
                    print(f"   {line.strip()}")
