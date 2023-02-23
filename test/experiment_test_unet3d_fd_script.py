import os
import glob
from tqdm import tqdm


def process_sentence_list(sentence_list, run=False):
    for sentence in tqdm(sentence_list):
         print(sentence)
         if run:
            os.system(sentence)

model_base_path = 'D:/program/MedicalZooPytorch-master/saved_models/heart_whole/'
hippocampus_percentage_list = ["10"]
test_sentence_list = []

for percentage in hippocampus_percentage_list:
    model_path_list = glob.glob(model_base_path+percentage+'/UNET3D_disentangle*.pth')
    for model_path in model_path_list:
        base_name = os.path.basename(model_path)
        model_name = base_name.split('-')[0].replace('_m','')
        modality_name = base_name.split('-')[1].replace('_a','')
        trained_model_base = model_base_path.split('/')[5]
        region = trained_model_base.split('_')[1]
        test_sentence = 'python inference_unet3d_fd.py --modalities='+modality_name+' --model='+model_name+' --cuda --classes=3 --regions='+region+' --save_prediction --fft_ratio=0.1 --trained_model_base_path=gnet_journal_experiment/'+trained_model_base+'/'+percentage+'/'+' --trained_model_name='+base_name.replace('.pth','')
        test_sentence_list.append(test_sentence)
process_sentence_list(test_sentence_list, False)