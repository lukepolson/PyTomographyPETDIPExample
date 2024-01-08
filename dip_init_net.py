from __future__ import annotations
import torch
from networks import UNetCustom
import pytomography
import numpy as np
import os
import misc
import simulation_parameters as P

net = UNetCustom([P.START_CHANNELS,2*P.START_CHANNELS,4*P.START_CHANNELS,8*P.START_CHANNELS,16*P.START_CHANNELS]).to(pytomography.device)
mri_aligned = torch.tensor(np.load(os.path.join(P.SAVE_PATH, 'mri_aligned.npy'))).unsqueeze(0).to(pytomography.device)
pet_recon = torch.tensor(np.load(os.path.join(P.SAVE_PATH, f'pet_recon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}.npy'))).unsqueeze(0).to(pytomography.device)

pipeline = misc.get_pipeline(mri_aligned, P.MRI_CROP_ABOVE, P.MRI_CROP_BELOW)

dip_prior = misc.DIPPrior(
    net,
    mri_aligned,
    pipeline,
    n_epochs=P.N_EPOCHS,
    scale_factor=P.MRI_SCALE_FACTOR,
    lr=P.LR_DIP_INIT_NET
    )
dip_prior.fit(pet_recon)

torch.save(dip_prior.network.state_dict(), os.path.join(P.SAVE_PATH, f'model_pet_recon_{P.N_EVENTS}events_osem_{P.N_ITERS_INIT}it{P.N_SUBSETS_INIT}ss{P.TOF}_initchannels{P.START_CHANNELS}'))
