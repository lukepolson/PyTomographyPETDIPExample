from __future__ import annotations
import numpy as np
import torch
import pytomography
from networks import UNetCustom
from pytomography.algorithms import DIPRecon
from pytomography.metadata import ObjectMeta, PETLMProjMeta
from pytomography.projectors import PETLMSystemMatrix
from pytomography.io.PET import get_scanner_LUT, gate
import os
import pickle
import misc
import simulation_parameters as P

# RECONSTRUCT
net = UNetCustom([P.START_CHANNELS,2*P.START_CHANNELS,4*P.START_CHANNELS,8*P.START_CHANNELS,16*P.START_CHANNELS]).to(pytomography.device)
net.load_state_dict(torch.load(os.path.join(P.SAVE_PATH, f'model_pet_recon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}_initchannels{P.START_CHANNELS}')))

# Load required information
mri_aligned = torch.tensor(np.load(os.path.join(P.SAVE_PATH, 'mri_aligned.npy'))).unsqueeze(0).to(pytomography.device)
pet_aligned = np.load(os.path.join(P.SAVE_PATH, 'pet_aligned.npy'))
mask_greymatter, mask_whitematter = misc.get_masks(P.PET_PHANTOM_PATH)
pet_recon = torch.tensor(np.load(os.path.join(P.SAVE_PATH, f'pet_recon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}.npy'))).unsqueeze(0).to(pytomography.device)

pipeline = misc.get_pipeline(mri_aligned, P.MRI_CROP_ABOVE, P.MRI_CROP_BELOW)

# Set initial image to pet_recon
dip_prior = misc.DIPPrior(
    net,
    mri_aligned,
    pipeline,
    n_epochs=0,
    scale_factor=P.MRI_SCALE_FACTOR,
    lr=P.LR_RECON
    )
dip_prior.fit(pet_recon)
dip_prior.n_epochs = P.N_SUBITERS2_DIP
dip_prior.max_iter = P.LBFGS_MAX_ITER

# RECONSTRUCT
detector_ids = torch.tensor(np.load(P.DETECTOR_IDS_PATH))
scanner_LUT = -torch.tensor(get_scanner_LUT(P.GEOMETRY_MACRO_PATH, mean_interaction_depth=9)).to(pytomography.dtype)

# Less events
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

if P.REMOVE_OOB:
    detector_ids = gate.remove_events_out_of_bounds(detector_ids, scanner_LUT, object_meta)

system_matrix = PETLMSystemMatrix(
    detector_ids,
    object_meta,
    proj_meta,
    obj2obj_transforms = [],
    attenuation_map=None,
    N_splits=8,
    eta_path = P.ETA_PATH,
    )

recon_algo = DIPRecon(
    projections=torch.tensor([1.]).to(pytomography.device),
    system_matrix=system_matrix,
    prior_network=dip_prior,
    rho=P.RHO,
)

cb = misc.StatisticsCallback(pet_aligned, mask_greymatter, mask_whitematter)
dip_final = recon_algo(P.N_ITERS_DIP,P.N_SUBITERS1_DIP, callback=cb)

np.save(os.path.join(P.SAVE_PATH, f'dip_recon_{P.N_ITERS_DIP}it_{P.N_SUBITERS1_DIP}subit1_{P.N_SUBITERS2_DIP}subit2_{P.RHO}rho_{P.LR_RECON}lr_{P.LBFGS_MAX_ITER}lbfgsiter_initpetrecon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}_initchannels{P.START_CHANNELS}'), dip_final[0].cpu().numpy())

with open(os.path.join(P.SAVE_PATH, f'dip_recon_{P.N_ITERS_DIP}it_{P.N_SUBITERS1_DIP}subit1_{P.N_SUBITERS2_DIP}subit2_{P.RHO}rho_{P.LR_RECON}lr_{P.LBFGS_MAX_ITER}lbfgsiter_initpetrecon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}_initchannels{P.START_CHANNELS}_callback'), 'wb') as f:
    pickle.dump(cb, f)