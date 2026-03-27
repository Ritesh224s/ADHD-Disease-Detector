"""
Create a professional feature engineering pipeline flowchart using matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Create figure and axis with high DPI for quality
fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
COLOR_INPUT = '#FFD700'      # Gold
COLOR_PROCESS = '#87CEEB'   # Sky blue
COLOR_OUTPUT = '#90EE90'    # Light green
COLOR_BORDER = '#333333'    # Dark gray
COLOR_ARROW = '#555555'     # Medium gray

# Function to create a rounded box with text
def create_box(ax, x, y, width, height, text, color, fontsize=11, fontweight='normal', linewidth=2.5):
    """Create a fancy box with rounded corners"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.15", 
        edgecolor=COLOR_BORDER, 
        facecolor=color, 
        linewidth=linewidth,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(x, y, text, 
            ha='center', va='center', 
            fontsize=fontsize,
            fontweight=fontweight,
            fontfamily='sans-serif',
            wrap=True,
            zorder=3)

# Function to create arrows with custom styling
def create_arrow(ax, x1, y1, x2, y2, width=2.5):
    """Create an arrow connecting two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', 
        mutation_scale=25,
        linewidth=width, 
        color=COLOR_ARROW,
        zorder=1
    )
    ax.add_patch(arrow)

# Create input node (top)
create_box(ax, 5, 10.5, 3.5, 1.2, 
           '19 Raw EEG Channels\n(Fp1 to O2)', 
           COLOR_INPUT, fontsize=12, fontweight='bold', linewidth=3)

# Create three processing nodes (middle layer)
box_y = 7
create_box(ax, 1.5, box_y, 2.8, 1.2,
           'Channel Statistics:\nMean & Std Dev\n(38 Features)',
           COLOR_PROCESS, fontsize=10)

create_box(ax, 5, box_y, 2.8, 1.2,
           'Global Statistics:\nMean, Range, Max,\nMin, Std\n(5 Features)',
           COLOR_PROCESS, fontsize=10)

create_box(ax, 8.5, box_y, 2.8, 1.2,
           'Spatial Interactions:\nTop 3 Variance\nChannel Products\n(3 Features)',
           COLOR_PROCESS, fontsize=10)

# Create output node (bottom)
create_box(ax, 5, 2.5, 3.5, 1.2,
           'Final Feature Space\n(65 Dimensions)',
           COLOR_OUTPUT, fontsize=12, fontweight='bold', linewidth=3)

# Create arrows from input to processing nodes
create_arrow(ax, 4.2, 9.9, 2.2, 7.6)
create_arrow(ax, 5, 9.9, 5, 7.6)
create_arrow(ax, 5.8, 9.9, 7.8, 7.6)

# Create arrows from processing nodes to output
create_arrow(ax, 1.5, 6.4, 3.5, 3.1)
create_arrow(ax, 5, 6.4, 5, 3.1)
create_arrow(ax, 8.5, 6.4, 6.5, 3.1)

# Add title
ax.text(5, 11.8, 'Feature Engineering Pipeline', 
        ha='center', va='center', 
        fontsize=16, fontweight='bold',
        fontfamily='sans-serif')

# Set background color
fig.patch.set_facecolor('white')

# Save with high quality
output_path = 'feature_engineering_flowchart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

# Verify the file was created
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"✓ Flowchart successfully created: {output_path}")
    print(f"✓ File size: {file_size / 1024:.2f} KB")
    print(f"✓ Resolution: 300 DPI (high quality)")
else:
    print("✗ Error: Failed to create flowchart")

plt.close()
