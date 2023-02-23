import torch.optim as optim

from .Unet3D import UNet3D
from .Unet3D_disentangle_early_fuse import Unet3D_disentangle_early_fuse
from .Unet3D_disentangle_late_fuse import Unet3D_disentangle_late_fuse
from .Unet3D_3 import UNet3D_3

model_list = ['UNET3D_3', 'UNET3D_disentangle_early_fuse', 'UNET3D_disentangle_late_fuse']


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'UNET3D_3':
        model = UNet3D_3(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'UNET3D_disentangle_early_fuse':
        model = Unet3D_disentangle_early_fuse(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'UNET3D_disentangle_late_fuse':
        model = Unet3D_disentangle_late_fuse(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)

    print(model_name, 'Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
