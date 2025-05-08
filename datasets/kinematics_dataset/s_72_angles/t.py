import scipy.io as sio
import numpy as np
data = sio.loadmat("S72_E2_A1.mat")
angles = data['angles']
angles_check = np.delete(angles, 5, axis=1)
print(np.isnan(angles_check).sum())
