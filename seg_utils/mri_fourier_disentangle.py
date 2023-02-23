import torch.fft
from parkinson_utils.preprocessing_total import *
from torch.fft import ifft2, fft2, fftshift, ifftn, fftn, ifftshift
import torch.nn.functional as F
from scipy.spatial import distance

width = 160
mask_width = 160*0.1
slice = 128
mask_slice = 128*0.1
mask_star_x = int(width / 2 - mask_width / 2)
mask_star_y = mask_star_x
mask_star_z = int(width / 2 - mask_width / 2)
mask_end_x = int(width / 2 + mask_width / 2)
mask_end_y = mask_end_x



def tensor_save_nii(mri_tensor, affine, save_path):
    mri_nii = nib.Nifti1Image(mri_tensor.numpy(), affine)
    nib.save(mri_nii, save_path)


def fourier_trans_vol_get_low_freq_tensor(mri_tensor):
    print('mri_tensor.size(),',mri_tensor.size())
    mri_tensor_fft = fftn(mri_tensor)
    mri_tensor_fft = fftshift(mri_tensor_fft)
    low_freq_fft = mri_tensor_fft[mask_star_x:mask_end_x, mask_star_y:mask_end_y]
    low_freq_ifft = ifftn(low_freq_fft)
    low_freq_ifft = torch.abs(low_freq_ifft)

    mri_tensor_fft[mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = 0
    high_freq_fft = mri_tensor_fft
    high_freq_ifft = ifftn(high_freq_fft)
    high_freq_ifft = torch.abs(high_freq_ifft)
    return low_freq_ifft, high_freq_ifft



def fourier_trans_vol_fda(mri_tensor1,mri_tensor2, beta):
    low_freq_tensor1,high_freq_tensor1 = fourier_trans_vol_get_low_freq_tensor(mri_tensor1)
    low_freq_tensor2,high_freq_tensor2 = fourier_trans_vol_get_low_freq_tensor(mri_tensor2)
    low_freq_tensor1_fftn = torch.fft.fftn(input=low_freq_tensor1)
    low_freq_tensor2_fftn = torch.fft.fftn(input=low_freq_tensor2)
    high_freq_tensor1_fftn = torch.fft.fftn(input=high_freq_tensor1)
    high_freq_tensor2_fftn = torch.fft.fftn(input=high_freq_tensor2)
    high_freq_tensor1_fftn[mask_star_x:mask_end_x, mask_star_y:mask_end_y,:] =low_freq_tensor2_fftn
    high_freq_tensor2_fftn[mask_star_x:mask_end_x, mask_star_y:mask_end_y,:] =low_freq_tensor1_fftn
    mri1to2 = torch.fft.ifftn(high_freq_tensor1_fftn).type(torch.float32)
    mri2to1 = torch.fft.ifftn(high_freq_tensor2_fftn).type(torch.float32)
    return mri1to2, mri2to1


save_base_path = "/Users/fuguanghui/Downloads/high_low_frequency/torch_nii/"

qsm_path = "/Users/fuguanghui/Downloads/high_low_frequency/QSM_AA_026.nii.gz"
r2star_path = "/Users/fuguanghui/Downloads/high_low_frequency/R2star_AA_026.nii.gz"
iMag_path = "/Users/fuguanghui/Downloads/high_low_frequency/iMag_AA_026.nii.gz"

qsm_tensor = path2tensor(qsm_path,'seg')
r2star_tensor = path2tensor(r2star_path,'seg')
iMag_tensor = path2tensor(iMag_path,'seg')

beta = 0.1
_, affine = path2np(qsm_path)
qsm_tensor_low, qsm_tensor_high = fourier_trans_vol_get_low_freq_tensor(qsm_tensor)
r2star_tensor_low, r2star_tensor_high = fourier_trans_vol_get_low_freq_tensor(r2star_tensor)
tensor_save_nii(qsm_tensor_low, affine, save_path=save_base_path+'qsm_low.nii.gz')
tensor_save_nii(qsm_tensor_high, affine, save_path=save_base_path+'qsm_high.nii.gz')
tensor_save_nii(r2star_tensor_low, affine, save_path=save_base_path+'r2star_low.nii.gz')
tensor_save_nii(r2star_tensor_high, affine, save_path=save_base_path+'r2star_high.nii.gz')

# mri1to2, mri2to1 = fourier_trans_vol_fda(qsm_tensor, r2star_tensor, beta)
# save_path_mri1to2 = save_base_path + 'qsm2r2star_torch_'+str(beta)+'.nii.gz'
# tensor_save_nii(mri1to2, affine, save_path_mri1to2)
# save_path_mri2to1 = save_base_path + 'r2star2qsm_torch_'+str(beta)+'.nii.gz'
# tensor_save_nii(mri2to1, affine, save_path_mri1to2)