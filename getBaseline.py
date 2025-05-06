import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Define joint names (update as needed based on your dataset's documentation)
joint_names = [
    'CMC1_F', 'CMC1_A', 'MCP1', 'IP1',
    'MCP2', 'PIP2', 'DIP2',
    'MCP3', 'PIP3', 'DIP3',
    'MCP4', 'PIP4', 'DIP4',
    'MCP5', 'PIP5', 'DIP5',
    'ABD', 'ROT', 'W_F', 'W_A',
    'W_R', 'W_U'
]

# Load data
mat = scipy.io.loadmat('kinematics_send/S2_E2_A1.mat', squeeze_me=True, struct_as_record=False)
angles = mat['angles']
movements = mat['re_stimulus'] if 're_stimulus' in mat else mat['stimulus']
movements = movements.squeeze()
repetitions = mat['re_repetition'] if 're_repetition' in mat else mat['repetition']
repetitions = repetitions.squeeze()

# Find indices
idx_movement_11 = np.where(movements == 11)[0]
idx_movement_10 = np.where(movements == 10)[0]

# Find the 'movement 0' between movement 10 and 11
if len(idx_movement_10) > 0 and len(idx_movement_11) > 0:
    last_10 = idx_movement_10[-1]
    first_11 = idx_movement_11[0]
    idx_movement_0_between_10_11 = np.where((movements == 0) & (np.arange(len(movements)) > last_10) & (np.arange(len(movements)) < first_11))[0]
else:
    idx_movement_0_between_10_11 = []

# Plot Movement 11 with repetition boundaries
plt.figure(figsize=(14, 6))
for joint_idx in range(angles.shape[1]):
    joint_label = joint_names[joint_idx] if joint_idx < len(joint_names) else f'Joint{joint_idx + 1}'
    plt.plot(angles[idx_movement_11, joint_idx], label=joint_label)

# Add vertical lines to mark repetition changes
movement_11_reps = repetitions[idx_movement_11]
unique_reps = np.unique(movement_11_reps)
for rep in unique_reps:
    rep_indices = np.where(movement_11_reps == rep)[0]
    if len(rep_indices) > 0:
        end_idx = rep_indices[-1]
        plt.axvline(x=end_idx, color='k', linestyle='--', alpha=0.5)

plt.title('Movement 11 - Full Duration of Joint Angles Over Time with Repetition Boundaries')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

