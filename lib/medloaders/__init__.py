from torch.utils.data import DataLoader
from .parkinson import Parkinson


def generate_datasets(args, path):
    params = {'batch_size': args.batchSz, 'shuffle': True, 'num_workers': 2}
    if args.dataset_name == "ParkinsonSeg":
        if args.run_mode == 'train':
            train_loader = Parkinson(args, 'train', dataset_path=path)
            val_loader = Parkinson(args, 'val', dataset_path=path)
            training_generator = DataLoader(train_loader, **params)
            val_generator = DataLoader(val_loader, **params)
            print("DATA SAMPLES for 'train' HAVE BEEN GENERATED SUCCESSFULLY")
            return training_generator, val_generator
        if args.run_mode == 'test_with_label':
            test_loader = Parkinson(args, 'test_with_label', dataset_path=path)
            test_generator = DataLoader(test_loader, **params)
            print("DATA SAMPLES for 'test_with_label' HAVE BEEN GENERATED SUCCESSFULLY")
            return test_generator
        if args.run_mode == 'test_without_label':
            test_loader = Parkinson(args, 'test_without_label', dataset_path=path)
            test_generator = DataLoader(test_loader, **params)
            print("DATA SAMPLES for 'test_without_label' HAVE BEEN GENERATED SUCCESSFULLY")
            return test_generator

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator
