import parkinson_utils.basic_util
import torch
import torchio as tio
import nibabel as nib
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import glob
import errno


def normalize_intensity(nii_np, normalization='max_min'):
    print('Normalization')
    MEAN, STD = nii_np.mean(), nii_np.std()
    MAX, MIN = nii_np.max(), nii_np.min()
    if normalization == "mean":
        mask = nii_np.ne(0.0)
        desired = nii_np[mask]
        mean_val, std_val = desired.mean(), desired.std()
        nii_np = (nii_np - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = np.max(nii_np)
        nii_np = nii_np / max_val
    elif normalization == 'brats':
        normalized_tensor = (nii_np.copy() - MEAN) / STD
        final_tensor = np.where(nii_np == 0., nii_np, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.copy() - MIN) / (MAX - MIN)) + 10.0
        x = np.where(nii_np == 0., nii_np, final_tensor)
        return x
    elif normalization == 'full_volume_mean':
        nii_np = (nii_np.copy() - MEAN) / STD
    elif normalization == 'max_min':
        nii_np = (nii_np - MIN) / ((MAX - MIN))
    elif normalization is None:
        nii_np = nii_np
    return nii_np

def path2nib(path):
    mri = nib.load(path)
    return mri


def path2np(path):
    mri_np = np.asarray(path2nib(path).get_fdata(dtype=np.float32))
    mri_affine = path2nib(path).affine
    return mri_np, mri_affine


def path2tensor(path, type):
    mri_nii = nib.load(path)
    mri_np = np.asarray(mri_nii.get_fdata(dtype=np.float32))
    mri_tensor = torch.from_numpy(mri_np)
    if type == 'mri':
        return mri_tensor.unsqueeze(0)
    else:
        return mri_tensor


def generate_mask(resolution, beta):
    mask_resolution = int(resolution * beta)
    mask = np.zeros(shape=(resolution, resolution))
    mask_star_x = int(resolution / 2 - mask_resolution / 2)
    mask_star_y = mask_star_x
    mask_end_x = int(resolution / 2 + mask_resolution / 2)
    mask_end_y = mask_end_x
    mask[mask_star_x:mask_end_x, mask_star_y:mask_end_y] = np.ones(shape=(mask_resolution))
    return mask


def fourier_trans(nii_np):
    nii_f = np.fft.fft2(nii_np)
    nii_fshift = np.fft.fftshift(nii_f)
    nii_res = np.log(np.abs(nii_fshift))
    return nii_fshift, nii_res


def mask_nii(nii_source_np, nii_target_np, mask_rate):
    mask = generate_mask(resolution=nii_source_np.shape[0], beta=mask_rate)
    nii_source_fshift, nii_source_res = fourier_trans(nii_source_np)
    nii_target_fshift, nii_target_res = fourier_trans(nii_target_np)
    nii_source_new_fshift = np.multiply(nii_target_fshift, mask) + np.multiply(nii_source_fshift, (1 - mask))
    nii_source_new_res = np.log(np.abs(nii_source_new_fshift))
    return nii_source_new_fshift, nii_source_new_res


def resize_mri_label(subject_name, mri_path, mri_save_path, label_path='', label_save_path='', target_shape=(160, 160, 128)):
    print('Processing Resize:', subject_name)
    try:
        transform = tio.transforms.Resize(target_shape=target_shape)
        mri_nii = tio.ScalarImage(mri_path)
        if label_path != '':
            label_nii = tio.ScalarImage(label_path)
            mri_subject = tio.Subject(image=mri_nii, label=label_nii)
            mri_nii_transform = transform(mri_subject)
            mri_nii_transform['image'].save(mri_save_path + subject_name)
            mri_nii_transform['label'].save(label_save_path + subject_name)
        else:
            mri_nii_transform = transform(mri_nii)
            mri_nii_transform.save(mri_save_path + subject_name)

    except:
        pass


def crop_pad_mri_label(subject_name, mri_path, mri_save_path, label_path='', label_save_path='', target_shape=(160, 160, 128)):
    print('Processing Crop or pad:', subject_name)
    try:
        transform = tio.transforms.CropOrPad(target_shape=target_shape)
        mri_nii = tio.ScalarImage(mri_path)
        if label_path != '':
            label_nii = tio.ScalarImage(label_path)
            mri_subject = tio.Subject(image=mri_nii, label=label_nii)
            mri_nii_transform = transform(mri_subject)
            mri_nii_transform['image'].save(mri_save_path + subject_name)
            mri_nii_transform['label'].save(label_save_path + subject_name)
        else:
            mri_nii_transform = transform(mri_nii)
            mri_nii_transform.save(mri_save_path + subject_name)

    except:
        pass



def save_nii(mri_np, mri_affine, save_path):
    mri_np = np.array(mri_np, dtype=np.float32)
    mri_nii = nib.Nifti1Image(mri_np, mri_affine)
    nib.save(mri_nii, save_path)


def dataset_statistics(base_path):
    file_list = os.listdir(base_path)
    width_total = []
    height_total = []
    slice_total = []
    labeled_width_total = []
    labeled_height_total = []
    labeled_slice_total = []
    index_total = []
    labeled_slice_slice_total = []
    category_amount = 0
    for file_name in file_list:
        mri_np, _ = path2np(base_path + file_name)
        mri_category = np.max(mri_np)
        if mri_category>category_amount:
            category_amount = mri_category
        labeled_slice_amount = 0
        slice_amount = mri_np.shape[2]
        slice_total.append(slice_amount)
        width_total.append(mri_np.shape[0])
        height_total.append(mri_np.shape[1])
        index_list = []
        for i in range(slice_amount):
            if np.sum(mri_np[..., i].flatten()) != 0.0:
                index_list.append(i)
                labeled_slice_amount += 1
        print('MRI name:', file_name, ', image size:', mri_np.shape, ', slice:', slice_amount, ', labeled slice:',
              labeled_slice_amount, '; start from No.', min(index_list), '; end to No.', max(index_list))
        index_total.append(min(index_list))
        index_total.append(max(index_list))
        labeled_slice_slice_total.append(labeled_slice_amount)
    print('Total data amount:', len(file_list),
          '\ndim: max (width, height, slice):', max(width_total), max(height_total), max(slice_total),
          '; min (width, height, slice):', min(width_total), min(height_total), min(slice_total), '; avg:',
          int(np.mean(width_total)), int(np.mean(height_total)), int(np.mean(slice_total)),
          '\nlabeled slice start from:', min(index_total), ', end to:', max(index_total),
          '\naverage labeled amount:', np.mean(labeled_slice_slice_total),
          ';\ncategory:',str(category_amount))



def index2list(scan_list, index_list):
    scan_list_new = []
    for index in index_list:
        scan_name = scan_list[index]
        scan_name = str(scan_name).replace('.nii.gz','')
        scan_list_new.append(scan_name)
    return scan_list_new

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_cv_index(scan_list, scan_label_list='', test_ratio=0.2, cv_fold=5, save_base_path=''):
    print('Split dataset, test:', test_ratio, ', cross validation:', cv_fold)
    mkdir(save_base_path)
    split_fuc = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, train_size=1 - test_ratio, random_state=42)
    if scan_label_list == '':
        scan_label_list = np.ones(shape=len(scan_list))
    for train_index, test_index in split_fuc.split(scan_list, scan_label_list):
        subject_train, subject_test = index2list(scan_list, train_index), index2list(scan_list, test_index)
        subject_train_label, subject_test_label = index2list(scan_label_list, train_index), index2list(scan_label_list, test_index)
        np.save(save_base_path + '/train.npy', np.array(subject_train))
        np.save(save_base_path + '/test.npy', np.array(subject_test))
        skf = StratifiedKFold(n_splits=cv_fold)
        i = 1
        assert len(subject_train) == len(subject_train_label)
        for train_index, val_index in skf.split(subject_train, subject_train_label):
            subject_cv_train, subject_cv_val = index2list(subject_train, train_index), index2list(subject_train, val_index)
            save_cv_base_path = save_base_path + '/cv/'
            parkinson_utils.basic_util.mkdir(save_cv_base_path)
            np.save(save_cv_base_path + str(i) + '_train.npy', np.array(subject_cv_train))
            np.save(save_cv_base_path + str(i) + '_val.npy', np.array(subject_cv_val))
            i += 1
