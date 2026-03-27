"""
EEG Topographic Brain Maps: Neurotypical vs ADHD Patient
Creates side-by-side toplot visualizations showing electrode activity
using standard 10-20 electrode layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from mne import create_info
from mne.channels import make_standard_montage
from mne.viz import plot_topomap
import warnings
warnings.filterwarnings('ignore')

# Standard 10-20 electrode names
electrode_names = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'Oz', 'O2'
]

# Create MNE info object with standard 10-20 montage
info = create_info(ch_names=electrode_names, sfreq=100, ch_types='eeg')
montage = make_standard_montage('standard_1020')
info.set_montage(montage)

# ============================================================================
# LEFT PLOT: Neurotypical (Control) - Normal Activity
# ============================================================================
# Create data showing normal, low activity (baseline = 0, slight variations)
np.random.seed(42)
control_data = np.random.normal(0.5, 0.3, len(electrode_names))
# Ensure all values are reasonable
control_data = np.clip(control_data, -1, 2)

# ============================================================================
# RIGHT PLOT: ADHD Patient - High Frontal Activity (Elevated Theta/Beta)
# ============================================================================
# Create data showing elevated activity, especially in frontal region
adhd_data = np.random.normal(2, 0.4, len(electrode_names))

# Enhance frontal regions (F3, Fz, F4, Fp1, Fp2, F7, F8)
frontal_indices = [
    electrode_names.index('Fp1'),
    electrode_names.index('Fp2'),
    electrode_names.index('F7'),
    electrode_names.index('F3'),
    electrode_names.index('Fz'),
    electrode_names.index('F4'),
    electrode_names.index('F8'),
]

for idx in frontal_indices:
    adhd_data[idx] = np.random.uniform(4.5, 6.0)  # High values for frontal region

# Also slightly elevate central region
central_indices = [
    electrode_names.index('C3'),
    electrode_names.index('Cz'),
    electrode_names.index('C4'),
]

for idx in central_indices:
    adhd_data[idx] = np.random.uniform(3.5, 4.5)

adhd_data = np.clip(adhd_data, 0, 7)

# Set common vmin/vmax for consistent color scaling
vmin = np.min([control_data.min(), adhd_data.min()])
vmax = adhd_data.max()

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot neurotypical brain map
img_control = plot_topomap(
    control_data,
    info,
    axes=axes[0],
    show=False,
    cmap='RdBu_r',
    vlim=(vmin, vmax),
    contours=4
)

axes[0].set_title('Neurotypical (Control)', fontsize=14, fontweight='bold', pad=10)

# Plot ADHD brain map
img_adhd = plot_topomap(
    adhd_data,
    info,
    axes=axes[1],
    show=False,
    cmap='RdBu_r',
    vlim=(vmin, vmax),
    contours=4
)

axes[1].set_title('ADHD Patient (High Theta/Beta)', fontsize=14, fontweight='bold', pad=10)

# ============================================================================
# Add overall title and adjust layout
# ============================================================================
fig.suptitle('EEG Topographic Maps: Standard 10-20 Electrode Layout', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# ============================================================================
# Save figure at 300 DPI
# ============================================================================
output_path = 'eeg_brain_map.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ EEG topographic map saved to: {output_path}")
print(f"  - Resolution: 300 DPI")
print(f"  - Left plot: Neurotypical (Control) - Normal activity")
print(f"  - Right plot: ADHD Patient - High frontal activity (Theta/Beta elevated)")
print(f"  - Electrode layout: Standard 10-20 ({len(electrode_names)} electrodes)")
print(f"  - Data range: {vmin:.2f} to {vmax:.2f} μV")

plt.close()
