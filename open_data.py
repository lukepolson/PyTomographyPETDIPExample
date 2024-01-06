from __future__ import annotations
import numpy as np
from pytomography.io.PET import gate
import simulation_parameters as P

paths = [f'/disk1/pet_mri_scan/gate_simulation/no_atten/f{i}.root' for i in range(1,19)]
detector_ids = gate.get_detector_ids(paths, P.GEOMETRY_MACRO_PATH, same_source_pos = P.SAME_SOURCE_POS)
np.save(P.DETECTOR_IDS_PATH, detector_ids.astype(np.int32))