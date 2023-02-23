import json
import os
import random
import shutil
import time
import pickle
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchio as tio


def reproducibility(args, seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # LET'S TEST IT
    cudnn.benchmark = True


def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()


def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)


def shuffle_lists(*ls, seed=777):
    l = list(zip(*ls))
    random.seed(seed)
    random.shuffle(l)
    return zip(*l)


def prepare_input(input_tuple, args=None):
    in_cuda = args.cuda
    run_mode = args.run_mode
    if run_mode == 'test_without_label':
        input_tensor = input_tuple[0]
        target = None
        if in_cuda:
            input_tensor = input_tensor.cuda()
        return input_tensor, target
    else:
        input_tensor, target = input_tuple
        return input_tensor, target


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)
    return list_file
