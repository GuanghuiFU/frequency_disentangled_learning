import os
import glob
from tqdm import tqdm


def train_gnet_sentence(task,region, classes, fft_ratio, batchsize, percentage, model, nEpochs=300):
    train_sentence = 'python train_parkinson.py --batchSz='+str(batchsize)+' --modalities_list=' + task + ' --regions='+region+' --val_modalities_list=' \
                         + task + ' --cuda --lr=1e-3 --nEpochs='+str(nEpochs)+' --classes='+str(classes)+' --percentage=' + str(percentage) + ' --model=' \
                         + model + ' --fft_ratio=' + str(fft_ratio)
    return train_sentence
def process_sentence_list(sentence_list, run=False, turn_off=False):
    for sentence in tqdm(sentence_list):
         print(sentence)
         if run:
            os.system(sentence)
    if turn_off:
        os.system("shutdown -s -t  60")
def generate_sentence(model_list, task_list, percentage_list, batchsize=1, fft_ratio=0.1):
    train_sentence_list = []
    for percentage in percentage_list:
        for task in task_list:
            task_name = task.split('_')[0]
            task_classes = task.split('_')[1]
            task_region = task.split('_')[2]
            for model in model_list:
                train_sentence = train_gnet_sentence(task_name, region=task_region, classes=task_classes, batchsize=batchsize, fft_ratio=fft_ratio, percentage=percentage, model=model)
                train_sentence_list.append(train_sentence)
    return train_sentence_list

def main():
    model_list = ['UNET3D_3', 'UNET3D_disentangle_early_fuse', 'UNET3D_disentangle_late_fuse']
    hippocampus_percentage_list = [2.5, 5, 10, 30, 50, 100] # number of training data: 260
    spleen_percentage_list = [15, 30, 100] # number of training data: 41
    heart_percentage_list = [30,100] # number of training data: 20
    task_list = ['hippocampus_3_whole', 'heart_2_whole', 'spleen_2_whole']
    hippocampus_train_list = generate_sentence(model_list[0:1], task_list[0:1], hippocampus_percentage_list[0:1])
    process_sentence_list(hippocampus_train_list, False)


if __name__ == '__main__':
    main()