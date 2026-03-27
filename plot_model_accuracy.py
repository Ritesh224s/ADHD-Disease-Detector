"""
Model Accuracy Comparison Bar Chart
Compares accuracy of different ML models for ADHD detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Model data
models = ['XGBoost', 'Gradient\nBoosting', 'Random\nForest', '1D CNN']
accuracies = [77.84, 75.00, 74.00, 72.00]
colors_list = ['#1b6b3e', '#2e86de', '#ff6348', '#ffa502']  # Green primary

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create bar chart
bars = ax.bar(models, accuracies, color=colors_list, edgecolor='black', linewidth=2.0, alpha=0.85)

# Add value labels on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
           f'{acc:.2f}%',
           ha='center', va='bottom', fontsize=13, fontweight='bold')

# Formatting
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('ADHD Classification Model Accuracy Comparison\nPredicted vs Actual Label Matching', 
            fontsize=15, fontweight='bold', pad=20)

# Set y-axis range
ax.set_ylim([60, 85])
ax.set_yticks(np.arange(60, 86, 5))

# Add grid
ax.yaxis.grid(True, alpha=0.4, linestyle='--')
ax.set_axisbelow(True)

# Add a horizontal line showing the primary model
ax.axhline(y=77.84, color='#1b6b3e', linestyle='--', linewidth=2, alpha=0.5, label='XGBoost Baseline')

# Highlight best model
ax.text(0, 79.5, '★ Selected Model', fontsize=11, fontweight='bold', color='#1b6b3e')

# Add performance insight
insight_text = ("Performance Ranking:\n" +
               "1. XGBoost: 77.84% ✓\n" +
               "2. Gradient Boost: 75.00%\n" +
               "3. Random Forest: 74.00%\n" +
               "4. 1D CNN: 72.00%")
ax.text(0.98, 0.50, insight_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='center', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', pad=1))

plt.tight_layout()

# Save figure
base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, 'model_accuracy_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Model Accuracy Comparison saved: {output_path}")

# Print summary
print("\n" + "="*70)
print("MODEL ACCURACY COMPARISON SUMMARY")
print("="*70)
for model, acc in zip(models, accuracies):
    model_clean = model.replace('\n', ' ')
    stars = "★" * int(acc / 10)
    print(f"{model_clean:20} {acc:6.2f}%  {stars}")
print("="*70)
print(f"\n✓ XGBoost selected as PRIMARY MODEL")
print(f"  Accuracy advantage over next best: {77.84 - 75.00:.2f}%")
print(f"  Accuracy advantage over CNN baseline: {77.84 - 72.00:.2f}%")
print("="*70 + "\n")

plt.close()
