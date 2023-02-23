import torch
import os
from parkinson_utils.preprocessing_total import save_nii, path2np, path2tensor
from torch.fft import ifftn, fftn, fftshift, ifftshift


def mask_tensor(input_tensor, beta=0.1):
    classes = input_tensor.size()[0]
    width = input_tensor.size()[1]
    slice = input_tensor.size()[3]
    mask_width = int(width * beta)
    mask_star_x = int(width / 2 - mask_width / 2)
    mask_star_y = mask_star_x
    mask_end_x = int(width / 2 + mask_width / 2)
    mask_end_y = mask_end_x
    mask = torch.zeros((classes, width, width, slice))
    mask[:,mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = torch.ones((classes, mask_width, mask_width, slice))
    input_fft_low_small = input_tensor[:,mask_star_x:mask_end_x, mask_star_y:mask_end_y, :]
    input_fft_high = torch.multiply(input_tensor, 1 - mask)
    input_fft_low_ori = torch.multiply(input_tensor, mask)
    return input_fft_low_small.squeeze(), input_fft_low_ori.squeeze(), input_fft_high.squeeze()

def nii_fourier_disentangle(shift, nii_path,save_path, file_name):
    _, affine = path2np(nii_path)
    nii_tensor = path2tensor(nii_path, type='mri')
    nii_tensor_f = fftn(nii_tensor)
    if shift: nii_tensor_f = fftshift(nii_tensor_f)
    input_fft_low_small, input_fft_low_ori, input_fft_high = mask_tensor(nii_tensor_f)
    if shift: input_fft_low_ori = ifftshift(input_fft_low_ori)
    input_fft_low_ori = ifftn(input_fft_low_ori)
    input_fft_low_ori = torch.abs(input_fft_low_ori)
    if shift: input_fft_high = ifftshift(input_fft_high)
    input_fft_high = ifftn(input_fft_high)
    input_fft_high = torch.abs(input_fft_high)
    if shift:
        save_nii(input_fft_low_ori, affine, save_path=save_path+'low_fftshift/'+file_name+'_low.nii.gz')
        save_nii(input_fft_high, affine, save_path=save_path+'high_fftshift/'+file_name+'_high.nii.gz')
    else:
        save_nii(input_fft_low_ori, affine, save_path=save_path + 'low/' + file_name + '_low.nii.gz')
        save_nii(input_fft_high, affine, save_path=save_path + 'high/' + file_name + '_high.nii.gz')

def main():
    example_path = "C:/Users/fugua/Downloads/visual_high_low/example/"
    save_base_path = "C:/Users/fugua/Downloads/visual_high_low/"
    example_nii_list = os.listdir(example_path)
    for example_nii_name in example_nii_list:
        print('MRI:', example_nii_name)
        nii_fourier_disentangle(shift=False, nii_path=example_path+example_nii_name, save_path=save_base_path, file_name=example_nii_name.replace('.nii.gz',''))
if __name__ == '__main__':
    main()