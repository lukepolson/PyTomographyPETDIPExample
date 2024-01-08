import numpy as np
import torch
import pytomography
import numpy.linalg as npl
from scipy.ndimage import affine_transform, binary_erosion
from torch.optim import LBFGS
import nibabel as nib
from pytomography.callbacks import Callback
from monai.transforms import ScaleIntensityd, CropForeground, Compose, DivisiblePadd, SpatialCropd, NormalizeIntensity, CropForegroundd, ScaleIntensityRangePercentilesd, InvertibleTransform, MapTransform, ThresholdIntensityd
import simulation_parameters as P

def align_highres_image(path, img=None):
    # If img is none, extract data from path
    data = nib.load(path)
    # If img is none, extract data from path
    if img is None:
        img = data.get_fdata()
    Sx, Sy, Sz = -(np.array(img.shape)-1) / 2
    dx, dy, dz = data.header['pixdim'][1:4]
    M_highres = np.zeros((4,4))
    M_highres[0] = np.array([dx, 0, 0, Sx*dx])
    M_highres[1] = np.array([0, dy, 0, Sy*dy])
    M_highres[2] = np.array([0, 0, dz, Sz*dz])
    M_highres[3] = np.array([0, 0, 0, 1])
    dx, dy, dz = P.dr
    Sx, Sy, Sz = -(np.array(P.shape)-1) / 2
    M_pet = np.zeros((4,4))
    M_pet[0] = np.array([dx, 0, 0, Sx*dx])
    M_pet[1] = np.array([0, dy, 0, Sy*dy])
    M_pet[2] = np.array([0, 0, dz, Sz*dz])
    M_pet[3] = np.array([0, 0, 0, 1])
    M = npl.inv(M_highres) @ M_pet
    return affine_transform(img, M, output_shape=P.shape, mode='constant', order=1)

def get_masks(pet_path):
    pet_data = nib.load(pet_path)
    pet_highres = pet_data.get_fdata()
    mask_greymatter = (pet_highres<P.GREY_MATTER_UPPER)*(pet_highres>P.GREY_MATTER_LOWER)
    mask_whitematter = (pet_highres<P.WHITE_MATTER_UPPER)*(pet_highres>P.WHITE_MATTER_LOWER)
    mask_whitematter = binary_erosion(mask_whitematter, iterations=1)
    mask_greymatter = binary_erosion(mask_greymatter, iterations=1)
    grey_matter_mask_aligned = align_highres_image(P.PET_PHANTOM_PATH, mask_greymatter.astype(float))>P.ALIGNMENT_OVERLAP
    white_matter_mask_aligned = align_highres_image(P.PET_PHANTOM_PATH, mask_whitematter.astype(float))>P.ALIGNMENT_OVERLAP
    return grey_matter_mask_aligned, white_matter_mask_aligned

def compute_CRC(img, pet_aligned, mask_greymatter, mask_whitematter):
    ar = img[mask_greymatter].mean()
    br = img[mask_whitematter].mean()
    atrue = pet_aligned[mask_greymatter].mean()
    btrue = pet_aligned[mask_whitematter].mean()
    return (ar/br-1)/(atrue/btrue-1)

def compute_mse(img, pet_aligned, mask):
    ratio = pet_aligned.sum() / img.sum()
    true_mean = pet_aligned[mask].mean()
    bias = (ratio*img - pet_aligned)[mask].mean()
    std = (ratio*img - pet_aligned)[mask].std()
    return bias/true_mean, std/true_mean

class StatisticsCallback(Callback):
    def __init__(
        self,
        pet_aligned,
        mask_greymatter,
        mask_whitematter
    ) -> None:
        self.pet_aligned = pet_aligned
        self.mask_greymatter = mask_greymatter
        self.mask_whitematter = mask_whitematter
        self.CRCs = []
        self.biass = []
        self.stds = []
        self.biass_wm = []
        self.stds_wm = []
        self.biass_gm = []
        self.stds_gm = []
    def run(self, object: torch.tensor, n_iter: int):
        CRC = compute_CRC(object.cpu().numpy()[0], self.pet_aligned, self.mask_greymatter, self.mask_whitematter)
        bias, std = compute_mse(object.cpu().numpy()[0], self.pet_aligned, None)
        bias_wm, std_wm = compute_mse(object.cpu().numpy()[0], self.pet_aligned, self.mask_whitematter)
        bias_gm, std_gm = compute_mse(object.cpu().numpy()[0], self.pet_aligned, self.mask_greymatter)
        print(f'Bias WM: {bias_wm}')
        print(f'std WM: {std_wm}')
        print(f'Bias GM: {bias_gm}')
        print(f'std GM: {std_gm}')
        self.CRCs.append(CRC)
        self.biass.append(bias)
        self.stds.append(std)
        self.biass_wm.append(bias_wm)
        self.stds_wm.append(std_wm)
        self.biass_gm.append(bias_gm)
        self.stds_gm.append(std_gm)
        
def get_pipeline(mri_aligned, mri_crop_above, mri_crop_below):
    roi_start, roi_end = CropForeground().compute_bounding_box(mri_aligned)
    pipeline = Compose([
        SpatialCropd(['MR', 'NM'], roi_start=roi_start, roi_end=roi_end, allow_missing_keys=True),
        DivisiblePadd(['MR', 'NM'], 16, allow_missing_keys=True),
        ThresholdIntensityd(['MR'], mri_crop_above, above=False, cval=mri_crop_above),
        ThresholdIntensityd(['MR'], mri_crop_below, above=True, cval=mri_crop_below),
        ScaleIntensityd(['MR'], 0, 1)
    ])
    return pipeline
    

class DIPPrior():
    def __init__(
        self,
        network,
        anatomical_image,
        pipeline,
        n_epochs=10,
        scale_factor=1,
        lr = 0.1,
    ):
        self.network = network
        self.anatomical_image = anatomical_image
        self.pipeline = pipeline
        self.n_epochs = n_epochs
        self.scale_factor = scale_factor
        self.lr = lr
        self.max_iter = 20
        
    def fit(self, object):
        data = self.pipeline({'NM': object, 'MR': self.anatomical_image})
        optimizer_lfbgs = LBFGS(self.network.parameters(), lr=self.lr, max_iter=self.max_iter, history_size=100)
        NM_truth = data['NM'].unsqueeze(0) * self.scale_factor
        network_input = data['MR'].unsqueeze(0)
        criterion = torch.nn.MSELoss()
        def closure(optimizer):
            optimizer.zero_grad()
            NM_prediction = self.network(network_input)
            loss = criterion(NM_prediction, NM_truth)
            loss.backward()
            return loss
        for epoch in range(self.n_epochs):  
            loss = optimizer_lfbgs.step(lambda: closure(optimizer_lfbgs))
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        self.network.zero_grad(set_to_none=True)
        with torch.no_grad():
            network_prediction = self.network(data['MR'].unsqueeze(0))[0]
        self.prior_object = self.pipeline.inverse({'NM': network_prediction})['NM'].as_tensor() / self.scale_factor
        
    def predict(self):        
        return self.prior_object.detach()