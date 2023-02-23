from parkinson_utils.preprocessing_total import *
from lib.visual3D_temp.viz import *
import seg_metrics.seg_metrics as sg
import pandas as pd



def eval2(pred_path, label_path, output_path):
    metrics = sg.write_metrics(labels=[1],  # exclude background
                               gdth_path=label_path,
                               pred_path=pred_path,
                               csv_file=output_path)
    #print(metrics)
def clean_filename(path):
    path_list = glob.glob(path+'/*.nii.gz')
    for file_path in path_list:
        basename = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        basename_clean = '_'.join(basename.split('_')[:-1])+'.nii.gz'
        os.rename(file_path,dir_path+'/'+basename_clean)
def cover_nii(path):
    path_list = glob.glob(path+'/*.nii.gz')
    for file_path in path_list:
        mri_np, affine = path2np(file_path)
        mri_np_new = np.where(mri_np==255,1,0)
        save_nii(mri_np_new, affine, file_path)
def cover_nii2(path):
    path_list = glob.glob(path+'/*.nii.gz')
    for file_path in path_list:
        mri_np, affine = path2np(file_path)
        mri_np_new = np.where(mri_np==255,1,mri_np)
        mri_np_new = np.where(mri_np_new==510,2,mri_np_new)
        save_nii(mri_np_new, affine, file_path)
def pandas_avg(path):
    print('*'*10, os.path.basename(path))
    path_list = glob.glob(path+'/eval.csv')
    for file_path in path_list:
        file_path = ''.join(file_path)
        eval_pd = pd.read_csv(file_path)
        temp = eval_pd[["dice", "jaccard", "precision", "recall", "fpr", "fnr", "vs", "hd", "msd", "mdsd", "stdsd", "hd95"]]
        eval_avg = temp.mean(axis=0)
        print(eval_avg)

def pandas_avg1(path):
    print('*'*10, os.path.basename(path))
    path_list = glob.glob(path+'/eval.csv')
    for file_path in path_list:
        file_path = ''.join(file_path)
        eval_pd = pd.read_csv(file_path)
        eval_pd = eval_pd.drop(eval_pd.index[[0, 20]])
        temp = eval_pd[["dice", "jaccard", "precision", "recall", "fpr", "fnr", "vs", "hd", "msd", "mdsd", "stdsd", "hd95"]]
        eval_avg = temp.mean(axis=0)
        print(eval_avg)

if __name__ == '__main__':
    pred_base_path = 'D:\\program\\MedicalZooPytorch-master\\saved_models\\gnet_journal_experiment_evaluation\\result\\qsm_redNucleus\\'
    rn_percentage_list = [7.5, 15, 30, 50, 100]  # RN
    hippocampus_percentage_list = [2.5, 5, 10, 30, 50, 100]  # 260
    pancreas_percentage_list = [2.5, 5, 10, 30, 50, 100]  # 279
    spleen_percentage_list = [15, 30, 100]  # 41
    prostate_percentage_list = [15, 30, 100]  # 32 files
    heart_percentage_list = [30, 100]  # 20
    percentage_list = rn_percentage_list

    task = pred_base_path.split("\\")[-2].split('_')[0]
    regions = pred_base_path.split("\\")[-2].split('_')[1]
    label_path = 'D:/Data/Parkinson/label/original/'+task+'/'+regions+'/'
    print('Label:',label_path)
    for percentage in percentage_list:
        modality_list = os.listdir(pred_base_path+str(percentage))
        for modality_folder in modality_list:
            pred_path = pred_base_path+str(percentage)+'/'+modality_folder+'/'
            print('Processing:', pred_path)
            #clean_filename(pred_path)
            #cover_nii2(pred_path)
            eval2(pred_path,label_path,pred_path+'/eval.csv')
            pandas_avg(pred_path)
