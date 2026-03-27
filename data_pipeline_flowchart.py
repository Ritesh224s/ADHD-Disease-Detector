"""
Data Preprocessing Pipeline Flowchart
Creates a vertical flowchart showing the EEG data preprocessing steps
using Matplotlib with academic styling.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
PRIMARY_COLOR = '#1565C0'
SECONDARY_COLOR = '#2E86AB'
LIGHT_BLUE = '#ADD8E6'
LIGHTER_BLUE = '#B3E5FC'
ARROW_COLOR = '#2E86AB'
BORDER_WIDTH = 2.5

# Function to draw rounded rectangle boxes
def draw_box(ax, x, y, width, height, text, fontsize=10, fontweight='normal', 
             fillcolor=LIGHT_BLUE, edgecolor=SECONDARY_COLOR, linewidth=2.5):
    """Draw a rounded rectangle box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor=edgecolor, facecolor=fillcolor,
                         linewidth=linewidth, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight=fontweight, fontfamily='Arial', wrap=True, zorder=4)

# Function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, width=2):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=width, color=ARROW_COLOR, zorder=2)
    ax.add_patch(arrow)

# ============================================================================
# Draw nodes (boxes)
# ============================================================================

# Node 1: Raw EEG Dataset
draw_box(ax, 5, 10.5, 4, 1, 
         'Raw EEG Dataset\n(2,166,383 samples)',
         fontsize=11, fontweight='bold', edgecolor=PRIMARY_COLOR)

# Node 2: Data Sanitization
draw_box(ax, 5, 8.8, 4.5, 1.1,
         'Data Sanitization\n(Remove Nulls, Handle NaNs)',
         fontsize=10, edgecolor=SECONDARY_COLOR)

# Node 3: Feature Normalization
draw_box(ax, 5, 7.1, 4.5, 1.1,
         'Feature Normalization\n(StandardScaler: z-score)',
         fontsize=10, edgecolor=SECONDARY_COLOR)

# Node 4: Stratified Data Split
draw_box(ax, 5, 5.4, 4.5, 1.1,
         'Stratified Data Split\n(Random State=42)',
         fontsize=10, fontweight='bold', edgecolor=SECONDARY_COLOR)

# Node 5a: Training Set
draw_box(ax, 2.5, 3.3, 3.5, 1,
         'Training Set\n(80%: 1,733,106 samples)',
         fontsize=10, fontweight='bold', fillcolor=LIGHTER_BLUE, edgecolor='#0277BD')

# Node 5b: Testing Set
draw_box(ax, 7.5, 3.3, 3.5, 1,
         'Testing Set\n(20%: 433,277 samples)',
         fontsize=10, fontweight='bold', fillcolor=LIGHTER_BLUE, edgecolor='#0277BD')

# ============================================================================
# Draw arrows (connections)
# ============================================================================

# Arrow from Node 1 to Node 2
draw_arrow(ax, 5, 10, 5, 9.35, width=2.5)

# Arrow from Node 2 to Node 3
draw_arrow(ax, 5, 8.35, 5, 7.65, width=2.5)

# Arrow from Node 3 to Node 4
draw_arrow(ax, 5, 6.65, 5, 5.95, width=2.5)

# Arrow from Node 4 to Node 5a (left split)
draw_arrow(ax, 4, 4.95, 3, 3.8, width=2.5)

# Arrow from Node 4 to Node 5b (right split)
draw_arrow(ax, 6, 4.95, 7, 3.8, width=2.5)

# ============================================================================
# Add title
# ============================================================================
ax.text(5, 11.7, 'Data Preprocessing Pipeline for EEG Analysis',
        ha='center', va='center', fontsize=14, fontweight='bold',
        fontfamily='Arial')

# Add subtitle with key information
ax.text(5, 1.8, 'Total Samples: 2,166,383 | Train/Test Split: 80/20 | Random State: 42',
        ha='center', va='center', fontsize=10, fontfamily='Arial',
        style='italic', color='#555555', bbox=dict(boxstyle='round', 
        facecolor='#f0f0f0', alpha=0.7, pad=0.5))

# ============================================================================
# Save the figure
# ============================================================================
output_file = 'data_pipeline_flowchart.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none')
print(f"✓ Data preprocessing pipeline flowchart created successfully!")
print(f"  - Output file: {output_file}")
print(f"  - Resolution: 300 DPI")
print(f"  - Style: Academic with light blue boxes")

print(f"✓ Data preprocessing pipeline flowchart created successfully!")
print(f"  - Output file: {output_file}")
print(f"  - Resolution: 300 DPI")
print(f"  - Style: Academic with light blue boxes")
print(f"\nPipeline structure:")
print(f"  1. Raw EEG Dataset (2,166,383 samples)")
print(f"     ↓")
print(f"  2. Data Sanitization (Remove Nulls, Handle NaNs)")
print(f"     ↓")
print(f"  3. Feature Normalization (StandardScaler: z-score)")
print(f"     ↓")
print(f"  4. Stratified Data Split (Random State=42)")
print(f"     ├→ Training Set (80%: 1,733,106 samples)")
print(f"     └→ Testing Set (20%: 433,277 samples)")

plt.close()
