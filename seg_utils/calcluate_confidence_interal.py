import pandas as pd
import numpy as np
import os
from scipy import stats


def cal_avg_confidence_interval(x):
    x_avg = np.average(x)
    x_std = x.std()
    x_ste = x_std/np.sqrt(len(x))
    interval = stats.norm.interval(0.95, x_avg, x_ste)
    return x_avg, x_ste, str(np.round(interval[0],2)), str(np.round(interval[1],2))


def summary_avg_confidence_interval(csv_path, txt_path):
    eval_pd = pd.read_csv(csv_path)
    dice_np = eval_pd[["dice"]].to_numpy()
    dice_avg, dice_ste, dice_conf_inter_low_bound, dice_conf_inter_upper_bound = cal_avg_confidence_interval(dice_np)
    precision_np = eval_pd[["precision"]].to_numpy()
    precision_avg, precision_ste, precision_conf_inter_low_bound, precision_conf_inter_upper_bound = cal_avg_confidence_interval(precision_np)
    recall_np = eval_pd[["recall"]].to_numpy()
    recall_avg, recall_ste, recall_conf_inter_low_bound, recall_conf_inter_upper_bound = cal_avg_confidence_interval(recall_np)
    hd95_np = eval_pd[["hd95"]].to_numpy()
    hd95_avg, hd95_ste, hd95_conf_inter_low_bound, hd95_conf_inter_upper_bound = cal_avg_confidence_interval(hd95_np)
    txt_df = pd.read_csv(txt_path,delimiter=",",header=None)
    num_row = txt_df.shape[0]
    pearsonr_list = txt_df.iloc[:num_row-1, 5]
    pearsonr_list_new = []
    for i in range(len(pearsonr_list)):
        pearsonr_i_new = pearsonr_list[i].replace(' Pearson\'s r:','')
        pearsonr_list_new.append(float(pearsonr_i_new))
    pearsonr_np = np.array(pearsonr_list_new)
    pearsonr_avg, pearsonr_ste, pearsonr_conf_inter_low_bound, pearsonr_conf_inter_upper_bound = cal_avg_confidence_interval(pearsonr_np)
    eval_value = []
    eval_item = ['Dice', 'HD95', 'Precision', 'Recall', 'Pearson r']
    eval_value.append(str(np.round(dice_avg*100,2))+' ['+dice_conf_inter_low_bound+','+dice_conf_inter_upper_bound+']')
    eval_value.append(str(np.round(hd95_avg,2)) + ' [' + hd95_conf_inter_low_bound + ',' + hd95_conf_inter_upper_bound + ']')
    eval_value.append(str(np.round(precision_avg*100,2))+' ['+precision_conf_inter_low_bound+','+precision_conf_inter_upper_bound+']')
    eval_value.append(str(np.round(recall_avg*100,2))+' ['+recall_conf_inter_low_bound+','+recall_conf_inter_upper_bound+']')
    eval_value.append(str(np.round(pearsonr_avg,2))+' ['+pearsonr_conf_inter_low_bound+','+pearsonr_conf_inter_upper_bound+']')
    for item, value in zip(eval_item, eval_value):
        print(item, ':', value)
    eval_value_str = '& '.join(eval_value)
    print('& ' + eval_value_str)


def summary_avg_standard_error(csv_path, txt_path):
    eval_pd = pd.read_csv(csv_path)
    dice_np = eval_pd[["dice"]].to_numpy()
    dice_avg, dice_ste, dice_conf_inter_low_bound, dice_conf_inter_upper_bound = cal_avg_confidence_interval(dice_np)
    precision_np = eval_pd[["precision"]].to_numpy()
    precision_avg, precision_ste, precision_conf_inter_low_bound, precision_conf_inter_upper_bound = cal_avg_confidence_interval(precision_np)
    recall_np = eval_pd[["recall"]].to_numpy()
    recall_avg, recall_ste, recall_conf_inter_low_bound, recall_conf_inter_upper_bound = cal_avg_confidence_interval(recall_np)
    hd95_np = eval_pd[["hd95"]].to_numpy()
    hd95_avg, hd95_ste, hd95_conf_inter_low_bound, hd95_conf_inter_upper_bound = cal_avg_confidence_interval(hd95_np)
    txt_df = pd.read_csv(txt_path,delimiter=",",header=None)
    num_row = txt_df.shape[0]
    pearsonr_list = txt_df.iloc[:num_row-1, 5]
    pearsonr_list_new = []
    for i in range(len(pearsonr_list)):
        pearsonr_i_new = pearsonr_list[i].replace(' Pearson\'s r:','')
        pearsonr_list_new.append(float(pearsonr_i_new))
    pearsonr_np = np.array(pearsonr_list_new)
    pearsonr_avg, pearsonr_ste, pearsonr_conf_inter_low_bound, pearsonr_conf_inter_upper_bound = cal_avg_confidence_interval(pearsonr_np)
    eval_value = []
    eval_item = ['Dice', 'HD95', 'Precision', 'Recall', 'Pearson r']
    eval_value.append(str(np.round(dice_avg*100,2))+'±'+str(np.round(dice_ste*100,2)))
    eval_value.append(str(np.round(hd95_avg,2)) + '±' + str(np.round(hd95_ste,2)))
    eval_value.append(str(np.round(precision_avg*100,2))+'±'+str(np.round(precision_ste*100,2)))
    eval_value.append(str(np.round(recall_avg*100,2))+'±'+str(np.round(recall_ste*100,2)))
    eval_value.append(str(np.round(pearsonr_avg,2))+'±'+str(np.round(pearsonr_ste,2)))
    for item, value in zip(eval_item, eval_value):
        print(item, ':', value)
    eval_value_str = '& '.join(eval_value)
    print('& ' + eval_value_str)


if __name__ == '__main__':
    search_base_path = 'D://program//MedicalZooPytorch-master//saved_models//gnet_journal_experiment_evaluation//result//qsm_redNucleus//'
    rn_percentage_list = [7.5, 15, 30, 50, 100]  # RN
    hippocampus_percentage_list = [2.5, 5, 10, 30, 50, 100]  # 260
    spleen_percentage_list = [15, 30, 100]  # 41
    heart_percentage_list = [30, 100]  # 20
    percentage_list = rn_percentage_list
    for percentage in percentage_list:
        search_folder_list = os.listdir(search_base_path+str(percentage)+'//')
        for folder_name in search_folder_list:
            folder_path = search_base_path+str(percentage)+'//'+folder_name+'/'
            print('*'*20, folder_path.replace(search_base_path,'').replace('high_backward', 'Late').replace('high_forward', 'Early'))
            csv_path = folder_path + "eval.csv"
            txt_path = folder_path + "evaluation.txt"
            summary_avg_confidence_interval(csv_path, txt_path)
