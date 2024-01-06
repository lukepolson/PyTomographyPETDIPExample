import numpy as np

# Save path
SAVE_PATH = '/disk1/pet_dip_recon'
ETA_PATH = '/disk1/pet_mri_scan/norm_factor_test.npy'
MRI_PHANTOM_PATH = '/disk1/pet_mri_scan/fdg_pet_phantom_mri.nii.gz'
PET_PHANTOM_PATH = '/disk1/pet_mri_scan/fdg_pet_phantom_uptake.nii.gz'
DETECTOR_IDS_PATH = '/disk1/pet_mri_scan/gate_simulation/no_atten/processed_ids_no_randoms.npy'
GEOMETRY_MACRO_PATH = '/disk1/pet_mri_scan/mMR_Geometry.mac'
# Normalization stuff
NORMALIZATION_CYLINDER_RADIUS = 320
# Initial Reconstruction
N_ITERS_INIT = 80
N_SUBSETS_INIT = 1
dr = (2, 2, 2)
shape = (128,128,96)
N_EVENTS = int(35e6) # -1 means all events
TOF = 'nonTOF'
SCANNER_DIAMETER = 2 * 348
N_TOF_BINS = 125
FWHM_TOF = 220 * 0.3 #ps to mm
TOF_BIN_EDGES = np.linspace(-1,1,N_TOF_BINS+1)*SCANNER_DIAMETER
TOF_BIN_WIDTH = TOF_BIN_EDGES[1] - TOF_BIN_EDGES[0]
TOF_N_SIGMAS = 3
# Initial Network Training
N_EPOCHS = 100
START_CHANNELS = 8
MRI_SCALE_FACTOR = 50
MRI_CROP_ABOVE = 250
MRI_CROP_BELOW = 120
LR_DIP_INIT_NET = 0.01
RHO = 5e4
# DIP RECON
LR_RECON = 1
N_ITERS_DIP = 100
N_SUBITERS1_DIP = 2
N_SUBITERS2_DIP = 10
LBFGS_MAX_ITER = 20
# MASKING 
GREY_MATTER_LOWER = 0.6
GREY_MATTER_UPPER = np.inf
WHITE_MATTER_LOWER = 0.15
WHITE_MATTER_UPPER = 0.4
ALIGNMENT_OVERLAP = 0.8

# Other 
TEST_LOAD_IDS = False
REMOVE_OOB = True
SAME_SOURCE_POS = True

