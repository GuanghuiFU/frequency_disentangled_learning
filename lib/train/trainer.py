import numpy as np
import torch
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from torch.fft import ifftn, fftn

# define function to create a mask tensor with given beta value
def mask_tensor(input_tensor, beta=0.1):
    # get tensor dimensions
    batch = input_tensor.size()[0]
    classes = input_tensor.size()[1]
    width = input_tensor.size()[2]
    slice = input_tensor.size()[4]
    # calculate mask dimensions
    mask_width = int(width * beta)
    mask_star_x = int(width / 2 - mask_width / 2)
    mask_star_y = mask_star_x
    mask_end_x = int(width / 2 + mask_width / 2)
    mask_end_y = mask_end_x
    # create a tensor with all zeros and size of input tensor
    mask = torch.zeros((batch, classes, width, width, slice))
    # set the region in the mask tensor to ones
    mask[:,:,mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = torch.ones((batch, classes, mask_width, mask_width, slice))
    # calculate the low and high frequency tensors using the mask
    input_fft_low = input_tensor[:,:,mask_star_x:mask_end_x, mask_star_y:mask_end_y, :]
    input_fft_high = torch.multiply(input_tensor, 1 - mask)
    return input_fft_low, input_fft_high

class Trainer:
    """
    Trainer class
    """
    def __init__(self, args, model, criterion, optimizer, train_data_loader, valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)
        self.nEpochs = args.nEpochs
        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):
            self.train_epoch(epoch)
            if self.do_validation:
                self.validate_epoch(epoch)
            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(self.args.save, epoch, val_loss, optimizer=self.optimizer)
            self.writer.write_end_of_epoch(epoch)
            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            if self.args.model == 'UNET3D_disentangle_early_fuse':
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                target = target.cuda()
                input_dim = tuple(range(2, input_tensor.ndim))
                input_f = fftn(input_tensor, dim=input_dim)
                input_low_f, input_high_f = mask_tensor(input_f)
                input_low_i = ifftn(input_low_f, dim=tuple(range(2, input_low_f.ndim))).type(torch.float32)
                input_high_i = ifftn(input_high_f, dim=tuple(range(2, input_high_f.ndim))).type(torch.float32)
                input_low_i, input_high_i = input_low_i.cuda(), input_high_i.cuda()
                input_low_i.requires_grad = True
                input_high_i.requires_grad = True
                output = self.model(input_low_i, input_high_i)
                loss, per_ch_score = self.criterion(output, target, epoch)
                loss.backward()
                self.optimizer.step()
                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'train', epoch * self.len_epoch + batch_idx)
            elif self.args.model == 'UNET3D_disentangle_late_fuse':
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                target = target.cuda()
                input_dim = tuple(range(2, input_tensor.ndim))
                input_f = fftn(input_tensor, dim=input_dim)
                input_low_f, input_high_f = mask_tensor(input_f)
                input_low_i = ifftn(input_low_f, dim=tuple(range(2, input_low_f.ndim))).type(torch.float32)
                input_high_i = ifftn(input_high_f, dim=tuple(range(2, input_high_f.ndim))).type(torch.float32)
                input_low_i, input_high_i = input_low_i.cuda(), input_high_i.cuda()
                input_low_i.requires_grad = True
                input_high_i.requires_grad = True
                output = self.model(input_low_i, input_high_i)
                loss, per_ch_score = self.criterion(output, target, epoch)
                loss.backward()
                self.optimizer.step()
                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'train', epoch * self.len_epoch + batch_idx)
            else:
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor, target = input_tensor.cuda(), target.cuda()
                input_tensor.requires_grad = True
                output = self.model(input_tensor)
                loss_dice, per_ch_score = self.criterion(output, target, epoch)
                loss_dice.backward()
                self.optimizer.step()
                self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train', epoch * self.len_epoch + batch_idx)
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            with torch.no_grad():
                if self.args.model == 'UNET3D_disentangle_early_fuse':
                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    target = target.cuda()
                    input_dim = tuple(range(2, input_tensor.ndim))
                    input_f = fftn(input_tensor, dim=input_dim)
                    input_low_f, input_high_f = mask_tensor(input_f)
                    input_low_i = ifftn(input_low_f, dim=tuple(range(2, input_low_f.ndim))).type(torch.float32)
                    input_high_i = ifftn(input_high_f, dim=tuple(range(2, input_high_f.ndim))).type(torch.float32)
                    input_low_i, input_high_i = input_low_i.cuda(), input_high_i.cuda()
                    input_low_i.requires_grad = False
                    input_high_i.requires_grad = False
                    output = self.model(input_low_i, input_high_i)
                    loss, per_ch_score = self.criterion(output, target, epoch)
                elif self.args.model == 'UNET3D_disentangle_late_fuse':
                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    target = target.cuda()
                    input_dim = tuple(range(2, input_tensor.ndim))
                    input_f = fftn(input_tensor, dim=input_dim)
                    input_low_f, input_high_f = mask_tensor(input_f)
                    input_low_i = ifftn(input_low_f, dim=tuple(range(2, input_low_f.ndim))).type(torch.float32)
                    input_high_i = ifftn(input_high_f, dim=tuple(range(2, input_high_f.ndim))).type(torch.float32)
                    input_low_i, input_high_i = input_low_i.cuda(), input_high_i.cuda()
                    input_low_i.requires_grad = False
                    input_high_i.requires_grad = False
                    output = self.model(input_low_i, input_high_i)
                    loss, per_ch_score = self.criterion(output, target, epoch)
                else:
                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    input_tensor, target = input_tensor.cuda(), target.cuda()
                    input_tensor.requires_grad = False
                    output = self.model(input_tensor)
                    loss, per_ch_score = self.criterion(output, target, epoch)
            self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val', epoch * self.len_epoch + batch_idx)
        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)
