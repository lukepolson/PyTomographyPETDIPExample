import numpy as np
from pytomography.io.PET import gate
import simulation_parameters as P

paths = [f'/disk1/pet_mri_scan/normalization_scan/mMR_Norm_{i}.root' for i in range(1,37)]
eta = gate.get_eta_cylinder_calibration(paths, P.GEOMETRY_MACRO_PATH, cylinder_radius=P.NORMALIZATION_CYLINDER_RADIUS)
np.save(P.ETA_PATH, eta)