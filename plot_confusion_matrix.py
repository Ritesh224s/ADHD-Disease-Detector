"""
Confusion Matrix Visualization for ADHD Classification Model
Displays True Positives, True Negatives, False Positives, False Negatives
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Hardcoded data from model results
TN = 160299  # True Negatives (Control correctly identified)
FP = 32807   # False Positives (Control wrongly identified as ADHD)
FN = 44003   # False Negatives (ADHD wrongly identified as Control)
TP = 109513  # True Positives (ADHD correctly identified)

# Create confusion matrix
cm = np.array([[TN, FP], 
               [FN, TP]])

# Calculate percentages
total = TN + FP + FN + TP
cm_percent = (cm / total) * 100

# Create figure with professional styling
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap with annotations
labels = ['Control', 'ADHD']
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='white')

# Add custom annotations with numbers and percentages
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        percent = cm_percent[i, j]
        # Add both count and percentage
        text = ax.text(j + 0.5, i + 0.5, f'{int(count):,}\n({percent:.1f}%)',
                      ha="center", va="center", color="black", fontsize=13, fontweight='bold')

# Set labels and title
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('ADHD Classification - Confusion Matrix\n(Accuracy: 77.84%, ROC-AUC: 0.8541)', 
            fontsize=16, fontweight='bold', pad=20)

# Add sensitivity and specificity information
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
plt.figtext(0.5, 0.02, 
           f'Sensitivity (True Positive Rate): {sensitivity:.1%}  |  Specificity (True Negative Rate): {specificity:.1%}',
           ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, 'confusion_matrix.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Confusion Matrix saved: {output_path}")

# Show summary statistics
print("\n" + "="*70)
print("CONFUSION MATRIX SUMMARY")
print("="*70)
print(f"\nTrue Negatives (TN):  {TN:>10,} ({TN/total*100:>6.2f}%) - Control correctly identified")
print(f"False Positives (FP): {FP:>10,} ({FP/total*100:>6.2f}%) - Control wrongly as ADHD")
print(f"False Negatives (FN): {FN:>10,} ({FN/total*100:>6.2f}%) - ADHD wrongly as Control")
print(f"True Positives (TP):  {TP:>10,} ({TP/total*100:>6.2f}%) - ADHD correctly identified")
print(f"\nTotal Test Samples:   {total:>10,}")
print(f"\nSensitivity (TPR):    {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"Specificity (TNR):    {specificity:.4f} ({specificity*100:.2f}%)")
print(f"Accuracy:             {(TP+TN)/total:.4f} ({(TP+TN)/total*100:.2f}%)")
print("="*70 + "\n")

plt.close()
