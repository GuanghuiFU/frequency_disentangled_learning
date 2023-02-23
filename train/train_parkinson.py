import argparse
import os
import sys
import torch
sys.path.insert(0, 'D:/program/MedicalZooPytorch-master/')
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
import lib.utils as utils
from lib.losses3D.dice import DiceLoss
import time

seed = 1777777


def main():
    args = get_arguments()
    argsDict = args.__dict__
    for k in args.__dict__:
        print(k + ":" + str(args.__dict__[k]))
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)
    with open(args.save + '.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    dataset_base_path = 'D:/Data/Parkinson/'

    training_generator, val_generator = medical_loaders.generate_datasets(args, path=dataset_base_path)
    model, optimizer = medzoo.create_model(args)

    if args.loss == 'dice':
        print('Loss function: Dice')
        criterion = DiceLoss(classes=args.classes)
    if args.pretrained != '':
        model.restore_checkpoint(args.pretrained)
    if args.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model = model.cuda()
    elif args.gpu > 1:
        print("Utilize ", torch.cuda.device_count(), " GPUs!")
        model = torch.nn.DataParallel(model).cuda()
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator, valid_data_loader=val_generator)
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default="ParkinsonSeg")
    parser.add_argument('--loader_type', type=str, default="full", choices=('full'))
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--inChannels', type=int, default=1) # Number of modalities you choose, see general.py, prepare_input()
    parser.add_argument('--inModalities', type=int, default=1) # Total modalities
    parser.add_argument('--percentage', type=str, default="100")

    parser.add_argument('--modalities_list', type=str, default='heart')
    parser.add_argument('--augmentation', type=str, default='')
    parser.add_argument('--augmentation_list', type=str, default='')
    parser.add_argument('--regions', type=str, default='whole')
    parser.add_argument('--val_modalities_list', type=str, default='heart')
    parser.add_argument('--val_augmentation', type=str, default='')
    parser.add_argument('--val_augmentation_list', type=str, default='')
    parser.add_argument('--test_modalities_list', type=str, default='')

    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to pretrained model (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D_3', choices=('UNET3D_3', 'UNET3D_disentangle_early_fuse', 'UNET3D_disentangle_late_fuse'))
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str, default='../runs/')
    parser.add_argument('--cross_validation', type=str, default="1")
    parser.add_argument('--dataset_base_path', type=str, default='')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--run_mode', type=str, default='train', choices=('train', 'test_with_label', 'test_without_label'))
    parser.add_argument('--matrix', type=str, default='dice')
    parser.add_argument('--loss', type=str, default='dice')

    parser.add_argument('--modalities_bank', type=str, default='')
    parser.add_argument('--fft_ratio', type=str, default='0.1')

    local_time = time.strftime('%Y%m%d-%H%M', time.localtime())
    args = parser.parse_args()
    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_m-' + args.modalities_list + '_a-' + args.augmentation + '_l-' + args.loss + '_c-' + args.cross_validation + '_' +'percen-' + args.percentage +'_'+ local_time

    if args.pretrained != '':
        args.save +='_p'
    return args

if __name__ == '__main__':
    main()
