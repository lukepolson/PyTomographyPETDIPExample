from __future__ import annotations
import numpy as np
import torch
import pytomography
from pytomography.metadata import ObjectMeta, PETLMProjMeta
from pytomography.projectors import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.io.PET import gate
from pytomography.transforms.shared import GaussianFilter
import os
import pickle
import time
import misc
import simulation_parameters as P

# Align GT phantoms with reconstruction parameters
mri_aligned = misc.align_highres_image(P.MRI_PHANTOM_PATH)
pet_aligned = misc.align_highres_image(P.PET_PHANTOM_PATH)
mask_greymatter, mask_whitematter = misc.get_masks(P.PET_PHANTOM_PATH)
np.save(os.path.join(P.SAVE_PATH, 'mri_aligned'), mri_aligned)
np.save(os.path.join(P.SAVE_PATH, 'pet_aligned'), pet_aligned)

# Load information
start = time.time()
if P.TEST_LOAD_IDS:
    paths = [f'/disk1/pet_mri_scan/gate_simulation/no_atten/f{i}.root' for i in range(1, 19)]
    detector_ids = torch.tensor(gate.get_detector_ids(paths, P.GEOMETRY_MACRO_PATH))
else:
    detector_ids = torch.tensor(np.load(P.DETECTOR_IDS_PATH))
scanner_LUT = -torch.tensor(gate.get_scanner_LUT(P.GEOMETRY_MACRO_PATH, mean_interaction_depth=9)).to(pytomography.dtype)
end = time.time()
print(f'Time to load detector ids: {end-start}')

# Less events / time of flight
detector_ids = detector_ids[0:int(P.N_EVENTS)]
if P.TOF=='nonTOF':
    detector_ids = detector_ids[:,:2]

object_meta = ObjectMeta(P.dr, P.shape)
proj_meta = PETLMProjMeta(scanner_LUT, tof=False)
proj_meta.num_tof_bins = P.N_TOF_BINS
proj_meta.tofbin_width = P.TOF_BIN_WIDTH
proj_meta.sigma_tof = torch.tensor([P.FWHM_TOF / 2.355], dtype=torch.float32) 
proj_meta.tofcenter_offset = torch.tensor([0], dtype=torch.float32)
proj_meta.nsigmas = P.TOF_N_SIGMAS
# Remove events OOB
if P.REMOVE_OOB:
    detector_ids = gate.remove_events_out_of_bounds(detector_ids, scanner_LUT, object_meta)

start = time.time()
system_matrix = PETLMSystemMatrix(
    detector_ids,
    object_meta,
    proj_meta,
    obj2obj_transforms = [],
    attenuation_map=None,
    N_splits=8,
    eta_path = P.ETA_PATH,
    device='cpu',
    )
end = time.time()
print(f'Time to compute system matrix: {end-start}')

cb = misc.StatisticsCallback(pet_aligned, mask_greymatter, mask_whitematter)
recon_algorithm = OSEM(
    projections=torch.tensor([1.]),
    system_matrix=system_matrix,
    device = 'cpu'
)

start = time.time()
recon = recon_algorithm(n_iters=P.N_ITERS_INIT, n_subsets=P.N_SUBSETS_INIT, callback=cb)
end = time.time()
print(f'Time to reconstruct: {end-start}')

np.save(os.path.join(P.SAVE_PATH, f'pet_recon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}'), recon[0].cpu().numpy())
with open(os.path.join(P.SAVE_PATH, f'pet_recon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}_callback'), 'wb') as f:
    pickle.dump(cb, f)