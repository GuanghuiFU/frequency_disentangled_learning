# Import necessary libraries
import os
import nibabel as nib
import numpy as np
import torch


# Define function to convert NIfTI file to PyTorch tensor
def path2tensor(path, type):
    mri_nii = nib.load(path) # Load NIfTI file using nibabel library
    mri_np = np.asarray(mri_nii.get_fdata(dtype=np.float32)) # Convert to numpy array
    mri_tensor = torch.from_numpy(mri_np) # Convert numpy array to PyTorch tensor
    if type == 'mri':
        return mri_tensor.unsqueeze(0) # Add extra dimension for channel if MRI data
    else:
        return mri_tensor # Otherwise, return the tensor as-is


# Define function to get file paths for MRI and segmentation data
def get_data_path(mode, mode_regions, mode_modalities_list, mode_augmentation, mode_augmentation_list, mode_root, cv_base_path, cross_validation_index, percentage='100'):
    mri_npy_path = []
    seg_npy_path = []
    print(mode, 'Modalities:', mode_modalities_list)
    if mode_modalities_list != '':
        modalities_list = str(mode_modalities_list).split(',')
        for modalities in modalities_list:
            # Load training data split for current modality and cross-validation index
            training_scan_list_npy = np.load(cv_base_path + modalities + '/cv/' + cross_validation_index + '_' + mode + '.npy')
            data_total_amount = len(training_scan_list_npy)
            percentage_amount = int(np.ceil(data_total_amount * float(percentage) * 0.01))
            training_scan_list_npy = training_scan_list_npy[:percentage_amount] # Subset data according to specified percentage
            for select_scan in training_scan_list_npy:
                # Construct file paths for MRI and segmentation data
                mri_path = os.path.join(mode_root, 'MRI/original/', modalities, select_scan) + '.nii.gz'
                mri_npy_path.append(mri_path)
                seg_path = os.path.join(mode_root, 'label/original/', modalities, mode_regions, select_scan) + '.nii.gz'
                seg_npy_path.append(seg_path)
    if mode_augmentation != '':
        print(mode, 'With Augmentation:', mode_augmentation)
        print(mode, 'Augmentation List:', mode_augmentation_list)
        augmentation_list = str(mode_augmentation_list).split(',')
        if mode_modalities_list == '':
            select_scan_list_npy = np.load(cv_base_path + 'default/cv/' + cross_validation_index + '_' + mode + '.npy')
        else:
            select_scan_list_npy = np.load(cv_base_path + modalities + '/cv/' + cross_validation_index + '_' + mode + '.npy')
        for augmentation_type in augmentation_list:
            for select_scan in select_scan_list_npy:
                # Construct file paths for augmented MRI and segmentation data
                augmentation_base_path = os.path.join(mode_root, 'MRI/augmentation/', mode_augmentation, augmentation_type, select_scan)
                augmentation_seg_base_path = os.path.join(mode_root, 'label/augmentation/', mode_augmentation, augmentation_type, mode_regions, select_scan)
                mri_path = augmentation_base_path + '.nii.gz'
                seg_path = augmentation_seg_base_path + '.nii.gz'
                mri_npy_path.append(mri_path)
                seg_npy_path.append(seg_path)
    return mri
