import argparse
import os
import numpy as np
import sys
import os
sys.path.insert(0, 'D:\program\MedicalZooPytorch-master')
import lib.utils as utils
import lib.medzoo as medzoo
import nibabel as nib
from lib.visual3D_temp import visualize_3D_no_overlap2, visualize_3D_no_overlap
from lib.medloaders.parkinson import path2tensor
from lib.utils.general import make_dirs

def write_summary(path, summary):
    with open(path + 'evaluation.txt', 'a+') as f:
        f.write(str(summary) + '\n')

def main():
    args = get_arguments()
    seed = 1777777
    utils.reproducibility(args, seed)
    trained_model_base_path = args.trained_model_base_path
    trained_model_name = args.trained_model_name
    modalities = args.modalities
    regions = args.regions
    epochs = args.epochs
    cross_validation_index = args.cross_validation_index


    if args.run_machine == 'laptop':
        dataset_base_path = '/Users/guanghui.fu/Downloads/Dataset/Parkinson/'
        cv_base_path = '/Users/guanghui.fu/Downloads/MedicalZooPytorch-master/cross_validation_index/parkinson/'+args.modalities

    if args.run_machine == 'icm':
        dataset_base_path = '/network/lustre/iss01/aramis/users/guanghui.fu/data/Parkinson/'
        cv_base_path = '/network/lustre/dtlake01/aramis/users/guanghui.fu/script/phd_seg/3d/MedicalZooPytorch-master/cross_validation_index/'+args.modalities
    if args.run_machine == 'pc':
        dataset_base_path = 'D:\\Data\\Parkinson\\'
        cv_base_path = 'D:\\program\\MedicalZooPytorch-master\\cross_validation_index\\parkinson\\'+args.modalities
    model, optimizer = medzoo.create_model(args)

    model.restore_checkpoint('../saved_models/'+trained_model_base_path+'/' + trained_model_name + '.pth')
    if args.cuda:
        model = model.cuda()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Model transferred in GPU.....")
    print('Model name: ', trained_model_name)
    args.save = os.path.join('../inference_checkpoints/test/', cross_validation_index, trained_model_name + '/')

    make_dirs(args.save)
    dice_list = []
    precision_list = []
    recall_list = []
    mver_list = []
    maver_list = []
    pearsonsr_list = []
    f1_list = []
    topo_2d_error_list = []
    topo_3d_error_list = []

    scan_list_npy_name = cv_base_path+'\\test.npy'
    scan_list_npy = np.load(scan_list_npy_name)
    for select_scan in scan_list_npy:
        mri_path = os.path.join(dataset_base_path, 'MRI/original/', modalities, select_scan) + '.nii.gz'
        seg_path = os.path.join(dataset_base_path, 'label/original/', modalities, regions, select_scan) + '.nii.gz'
        #print('MRI:', mri_path, ', Segmentation:', seg_path)
        mri_tensor = path2tensor(mri_path, 'mri')
        mri_tensor = mri_tensor.unsqueeze(0)
        seg_tensor = path2tensor(seg_path, 'seg')
        if args.cuda:
            mri_tensor = mri_tensor.cuda()
            #seg_tensor = seg_tensor.cuda()
        summary, dice, precision, recall, mver, maver, pearsonsr, f1, topo_2d_err, topo_3d_err = visualize_3D_no_overlap(args=args, filename=select_scan, num_classes=args.classes, full_volume=mri_tensor, seg_map=seg_tensor, model=model,affine=nib.load(mri_path).affine)

        dice_list.append(dice)
        precision_list.append(precision)
        recall_list.append(recall)
        mver_list.append(mver)
        maver_list.append(maver)
        pearsonsr_list.append(pearsonsr)
        f1_list.append(f1)
        topo_2d_error_list.append(topo_2d_err)
        topo_3d_error_list.append(topo_3d_err)
        write_summary(args.save, summary)
    avg_dice = np.average(np.array(dice_list))
    avg_precision = np.average(np.array(precision_list))
    avg_recall = np.average(np.array(recall_list))
    avg_mver = np.average(np.array(mver_list))
    avg_maver = np.average(np.array(maver_list))
    avg_pearsonsr = np.average(np.array(pearsonsr_list))
    avg_f1 = np.average(np.array(f1_list))
    avg_topo_2d = np.average(np.array(topo_2d_error_list))
    avg_topo_3d = np.average(np.array(topo_3d_error_list))
    final_summary = 'Average Dice score: '+ np.array2string(avg_dice) + ', Precision:'+ np.array2string(avg_precision) + ', Recall:'+ np.array2string(avg_recall) + ', MVER:'+ np.array2string(avg_mver) + ', MAVER:'+ np.array2string(avg_maver) + ', Pearson\'s r:'+ np.array2string(avg_pearsonsr) + ', F1:'+ np.array2string(avg_f1) + '; 2D Topology error:'+ np.array2string(avg_topo_2d) + '; 3D Topology error:'+ np.array2string(avg_topo_3d)
    print(final_summary)
    write_summary(args.save, final_summary)
    new_path = os.path.join('../inference_checkpoints/test/', cross_validation_index, trained_model_name + '_' + str(epochs) + 'e_' + str(np.around(avg_dice, 4)))
    os.rename(args.save, new_path)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="ParkinsonSeg")
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', type=str, default='UNET3D', choices=('VNET', 'VNET2', 'UNET3D', 'UNET3D_3', 'UNET3D_Fourier','UNET3D_Fourier_Mul','DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', 'DENSEVOXELNET'))
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--run_machine', type=str, default='pc', choices=('laptop', 'icm', 'pc'))
    parser.add_argument('--run_mode', type=str, default='test_with_label', choices=('train', 'test_with_label', 'test_without_label', 'val'))
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--trained_model_name', type=str, default="UNET_qsm_cv1")
    parser.add_argument('--trained_model_base_path', type=str, default="UNET_qsm_cv1")
    parser.add_argument('--modalities', type=str, default="synthesis")
    parser.add_argument('--regions', type=str, default="3")
    parser.add_argument('--epochs', type=str, default="100")
    parser.add_argument('--cross_validation_index', type=str, default="1")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
