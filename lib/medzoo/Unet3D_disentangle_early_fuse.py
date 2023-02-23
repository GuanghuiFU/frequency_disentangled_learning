import torch.nn as nn
import torch
# from torchsummary import summary
# import torchsummaryX
from lib.medzoo.BaseModelClass import BaseModel
from pytorch_model_summary import summary
from torch.fft import fftn, ifftn


def merge_low_high(tensor_low, tensor_high, beta=0.1):
    # print('tensor_low.size():', tensor_low.size())
    # print('tensor_high.size():', tensor_high.size())
    width = tensor_high.size()[2]
    mask_width = int(width * beta)
    mask_star_x = int(width / 2 - mask_width / 2)
    mask_star_y = mask_star_x
    mask_end_x = int(width / 2 + mask_width / 2)
    mask_end_y = mask_end_x
    # print('merge_low_high: mask_star_x:mask_end_x, mask_star_y:mask_end_y:', mask_star_x, mask_end_x, mask_star_y, mask_end_y)
    tensor_high[:, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = tensor_low
    return tensor_high


class Unet3D_disentangle_early_fuse(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(Unet3D_disentangle_early_fuse, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_low = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)


        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x_low_i, x_high_i):
        # print('forward x_high_i.size():', x_high_i.size())
        output_high_i = self.conv3d_c1_1(x_high_i)
        # print('forward output_high_i:',output_high_i.size())
        output_low_i = self.conv3d_c1_low(x_low_i)
        # print('forward output_low_i:', output_low_i.size())
        output_low_f = fftn(output_low_i, dim=tuple(range(2, output_low_i.ndim)))
        output_high_f = fftn(output_high_i, dim=tuple(range(2, output_high_i.ndim)))
        output_low_high_f = merge_low_high(output_low_f, output_high_f)
        # print('forward output_low_high_f.size():', output_low_high_f.size())
        output_low_high_i = ifftn(output_low_high_f, dim=tuple(range(2, output_low_high_f.ndim))).type(torch.float32)
        # print('out_low_high conv3dl2 & 2:', out_low_high.shape)
        out = output_low_high_i
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        # print('level 1, out.shape:', out.shape)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        # print('level 2, out.shape:', out.shape)

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out
        # print('level 3, out.shape:', out.shape)

        # Level 1 localization pathway
        out = torch.cat([out, context_3], dim=1)
        # print('out cat:', out.shape)
        out = self.conv_norm_lrelu_l2(out)
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        # print('out conv3dl1:', out.shape)

        # Level 2 localization pathway
        out = torch.cat([out, context_2], dim=1)
        # # print('out conv3dl1 & 3:', out.shape)
        out = self.conv_norm_lrelu_l3(out)
        out = self.conv3d_l3(out)
        # # print('out conv3dl2:', out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        # print('out norm_lrelu_upscale_conv_norm_lrelu_l3:', out.shape)
        out = torch.cat([out, context_1], dim=1)
        # print('out_low_high conv3dl2 & 2:', out_low_high.shape)
        out = self.conv_norm_lrelu_l4(out)
        ds3 = out
        out = self.conv3d_l4(out)
        # print('out_low_high conv3dl3:', out_low_high.shape)
        out = out
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        out = out + ds3_1x1_conv
        # print('out_low_high.size():', out_low_high.size())
        #print('seg_layer_tensor_list.size():', seg_layer_tensor_list.size())
        return out

    def test(self, device='cpu'):
        num_sample_batch = 2
        input_tensor_low1 = torch.rand(num_sample_batch, self.n_classes, 16, 16, 128)
        input_tensor_low2 = torch.rand(num_sample_batch, self.n_classes, 16, 16, 128)
        input_tensor_low3 = torch.rand(num_sample_batch, self.n_classes, 16, 16, 128)
        input_tensor_low_tensor_list = torch.stack([input_tensor_low1, input_tensor_low2, input_tensor_low3], dim=0)
        # print('input_tensor_low_tensor_list.size():', input_tensor_low_tensor_list.size())
        input_tensor_high = torch.rand(num_sample_batch, self.n_classes, 160, 160, 128)
        ideal_out = torch.rand(3, num_sample_batch, self.n_classes, 160, 160, 128)
        out = self.forward(input_tensor_low_tensor_list, input_tensor_high)
        assert ideal_out.shape == out.shape, "'ideal_out' and 'out' must have the same shape"
        # summary(self.to(torch.device(device)), (2, 160, 160, 128), device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        # # print("Unet3D test is complete")


if __name__ == '__main__':
    unet_fftn = Unet3D_disentangle_early_fuse(in_channels=2, n_classes=2)
    unet_fftn.test()
    #summary(unet, input_tensor, dtypes=[torch.float32], device=torch.device('cuda'))
    ## print(summary(unet_fft2, tuple((input_tensor_low, input_tensor_high)), show_input=True, show_hierarchical=False))
    #     # print(summary(unet_fft, tuple([input_tensor_low, input_tensor_high]), show_input=True, show_hierarchical=False))
