import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.convlstm import ConvLSTM, ConvLSTMCell

try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class TMB(nn.Module):
    def __init__(self):
        super(TMB, self).__init__()
        self.t_process = nn.Sequential(*[
            nn.Conv2d(1, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        ])
        self.f_process = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ])

    def forward(self, x, t):
        feature = self.f_process(x)
        modulation_vector = self.t_process(t)
        output = feature * modulation_vector
        return output


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8, use_time=False):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if use_time == True:
            self.TMB_A_l1 = TMB()
            self.TMB_B_l1 = TMB()
            self.TMB_A_l2 = TMB()
            self.TMB_B_l2 = TMB()
            self.TMB_A_l3 = TMB()
            self.TMB_B_l3 = TMB()

    def forward(self, fea1, fea2, t=None, t_back=None):
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset)) if t is None else self.lrelu(
            self.L3_offset_conv2_1(L3_offset)) + self.TMB_A_l3(L3_offset, t)
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        B, C, L2_H, L2_W = fea1[1].size()
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset)) if t is None else self.lrelu(
            self.L2_offset_conv3_1(L2_offset)) + self.TMB_A_l2(L2_offset, t)
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        B, C, L1_H, L1_W = fea1[0].size()
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset)) if t is None else self.lrelu(
            self.L1_offset_conv3_1(L1_offset)) + self.TMB_A_l1(L1_offset, t)
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset)) if t_back is None else self.lrelu(
            self.L3_offset_conv2_2(L3_offset)) + self.TMB_B_l3(L3_offset, t_back)
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset)) if t_back is None else self.lrelu(
            self.L2_offset_conv3_2(L2_offset)) + self.TMB_B_l2(L2_offset, t_back)
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset)) if t_back is None else self.lrelu(
            self.L1_offset_conv3_2(L1_offset)) + self.TMB_B_l1(L1_offset, t_back)
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)
        return y


class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        try:
            L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        except RuntimeError:
            L3_fea = L3_fea.view(B, N, -1, L3_fea.shape[2], L3_fea.shape[3])

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(), L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(), L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
        return fusion_fea


class DeformableConvLSTM(ConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        ConvLSTM.__init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                          batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        #### extract features (for each frame)
        nf = input_dim

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_tensor, hidden_state=None):
        '''
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None.

        Returns
        -------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, t, :, :, :]
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor,
                                                 cur_state=[h_temp, c_temp])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        return super()._init_hidden(batch_size, tensor_size)


class BiDeformableConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim,
                                              kernel_size=kernel_size, num_layers=num_layers, front_RBs=front_RBs,
                                              groups=groups, batch_first=batch_first, bias=bias,
                                              return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2 * input_dim, input_dim, 1, 1, bias=True)

    def forward(self, x):
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        rev_rev = out_rev[0][:, reversed_idx, ...]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B * N, -1, H, W)
        result = self.conv_1x1(result)
        return result.view(B, -1, C, H, W)


# Model in each expert (Deformable Version)
"""
Two approaches of each expert's bottleneck network
1 -------
        |-----fea---  |--------
2 -------             |       |
        |-----fea---  |-------+----------Interpolated result
3--------             |       |
        |-----fea---  |--------
4--------             |
                      | 
GateNetwork  ----------

===============Without Neighboring Alignment and Deformable (Adopted Version)===========

1 ------------fea-----|--------
                      |         |
2 ------------fea-----|--------
                      |         +--------Interpolated result
3 ------------fea-----|--------
                      |         |
4-------------fea-----|--------
                      | 
GateNetwork  ----------

"""


class FeatureExtractionLR(nn.Module):
    def __init__(self, nf=64):
        super(FeatureExtractionLR, self).__init__()
        # deformable
        self.offset_conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3,
                                      stride=1, padding=1, bias=True
                                      )
        self.offset_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3,
                                      stride=1, padding=1, bias=True
                                      )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.Lr_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
        #                              deformable_groups=groups)

    def forward(self, x):
        """
        f1, fe: [B, C, H, W] features
        """
        LR_offset = x
        # B, N, C, H, W = Lr_offset.size()
        LR_offset = self.lrelu(self.offset_conv1(LR_offset))
        LR_offset = self.lrelu(self.offset_conv2(LR_offset))
        Lr_fea = LR_offset
        return Lr_fea


class GateNetwork1(nn.Module):
    def __init__(self, batch_size=6, nframes=4, img_size=32):
        super(GateNetwork1, self).__init__()
        self.batch_size = batch_size
        self.nframes = nframes
        self.img_size = img_size
        self.nf = self.batch_size * self.nframes
        self.conv3d = nn.Conv3d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.global_avg_pooling = nn.AvgPool2d(self.img_size)
        self.conv1d = nn.Conv1d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, bias=True)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Softmax = nn.Softmax(1)

    def forward(self, x):
        # x_dimension input should be getting feature extracted map with size [ 24, 64, 32, 32]
        weight_prob = self.lrelu(self.conv3d(x))
        weight_prob = self.global_avg_pooling(weight_prob)  # [24, 3, 1, 1]
        weight_prob = weight_prob.squeeze()  # [24, 3]
        weight_prob = self.lrelu(self.conv1d(weight_prob))
        weight_prob = weight_prob.squeeze()  # [24]
        weight_prob = weight_prob.view(-1, self.nframes)  # [6, 4]
        weight_prob = self.Softmax(weight_prob)
        return weight_prob


class GateNetwork2(nn.Module):
    def __init__(self, batch_size=6, nframes=4, img_size=32):
        super(GateNetwork2, self).__init__()
        self.batch_size = batch_size
        self.nframes = nframes
        self.img_size = img_size
        self.nf = self.batch_size * self.nframes
        self.conv3d = nn.Conv3d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.global_avg_pooling = nn.AvgPool2d(self.img_size)
        self.conv1d = nn.Conv1d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, bias=True)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Softmax = nn.Softmax(1)

    def forward(self, x):
        # x_dimension input should be getting feature extracted map with size [ 24, 64, 32, 32]
        weight_prob = self.lrelu(self.conv3d(x))
        weight_prob = self.global_avg_pooling(weight_prob)  # [24, 3, 1, 1]
        weight_prob = weight_prob.squeeze()  # [24, 3]
        weight_prob = self.lrelu(self.conv1d(weight_prob))
        weight_prob = weight_prob.squeeze()  # [24]
        weight_prob = weight_prob.view(-1, self.nframes)  # [6, 4]
        weight_prob = self.Softmax(weight_prob)
        return weight_prob


class GateNetwork3(nn.Module):
    def __init__(self, batch_size=6, nframes=4, img_size=32):
        super(GateNetwork3, self).__init__()
        self.batch_size = batch_size
        self.nframes = nframes
        self.img_size = img_size
        self.nf = self.batch_size * self.nframes
        self.conv3d = nn.Conv3d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.global_avg_pooling = nn.AvgPool2d(self.img_size)
        self.conv1d = nn.Conv1d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, bias=True)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Softmax = nn.Softmax(1)

    def forward(self, x):
        # x_dimension input should be getting feature extracted map with size [ 24, 64, 32, 32]
        weight_prob = self.lrelu(self.conv3d(x))
        weight_prob = self.global_avg_pooling(weight_prob)  # [24, 3, 1, 1]
        weight_prob = weight_prob.squeeze()  # [24, 3]
        weight_prob = self.lrelu(self.conv1d(weight_prob))
        weight_prob = weight_prob.squeeze()  # [24]
        weight_prob = weight_prob.view(-1, self.nframes)  # [6, 4]
        weight_prob = self.Softmax(weight_prob)
        return weight_prob

class SparseDispatcher(object):
    """
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, device):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()  # [6, 6, 6, 6]
        # print(f"part size is: {self._part_sizes}")
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]  # [24, 4]
        # print(f"expanding gates to match: {gates_exp.size()}")
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  # [24, 1]
        # print(f"non_zero_gates {self._nonzero_gates.size()}")
        self.device = device

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        B, N, C, H, W = inp.size()
        inp = inp.view(-1, C, H, W)
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        result = torch.split(inp_exp, self._part_sizes, dim=0)
        # print(f"Result size is: {result[0].size()}")
        return result, self._nonzero_gates

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        expert_cat = torch.cat(expert_out, dim=0)
        original_size = expert_cat.size()
        expert_cat = expert_cat.view(self._nonzero_gates.size()[0], -1)
        # print(expert_cat.size())

        # multipy softmax value to each index image
        if multiply_by_gates:
            expert_cat = expert_cat.mul(self._nonzero_gates)
            print(f"After multiplication {expert_cat.size()}")  # [24, 65536]
        expert_cat = expert_cat.view(-1, self._num_experts, 64, 32, 32)
        # print(expert_cat.size())
        # concatenate each frame
        result = torch.zeros(6, 64, 32, 32).to(self.device) # need to redo the code here
        # print(f"Initiate result: {result.size()}")
        for i in range(0, self._num_experts, 2):
            # can also use torch.cat with squeeze
            result += torch.add(expert_cat[:, i, :, :, :].clone(), expert_cat[:, i + 1, :, :, :])

        # print(result.size())
        # final result should have dimension [6, 64, 32, 32]
        return result

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        result = torch.split(self._nonzero_gates, self._part_sizes, dim=0)
        # print(f"Expert to gates result: {result[0].size()}")
        return result


class NextMoENet(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10,  opt=None, device=None):
        super(NextMoENet, self).__init__()
        self.opt = opt
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        self.device = device
        p_size = 48  # a place holder, not so useful
        patch_size = (p_size, p_size)
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        # ==============Feature Extraction HEAD Block=======================
        NeXtBlock_f = functools.partial(mutil.ConvNextBlock_STVSR, nf=nf)
        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        # self.feature_extraction = mutil.make_layer(NeXtBlock_f, front_RBs)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)


        # ===============Frame Interpolation (Mixture of Expert)=============

        self.num_experts = self.in_frames
        self.FeatureExtractionLR = FeatureExtractionLR(nf=self.nf)

        # Instantiate Experts
        self.experts = nn.ModuleList([self.FeatureExtractionLR for i in range(self.num_experts)])
        # GateNetwork
        self.gate1 = GateNetwork1(batch_size=opt["datasets"]["train"]["batch_size"], nframes=4, img_size=32)
        self.gate2 = GateNetwork2(batch_size=opt["datasets"]["train"]["batch_size"], nframes=4, img_size=32)
        self.gate3 = GateNetwork3(batch_size=opt["datasets"]["train"]["batch_size"], nframes=4, img_size=32)
        # ===============ConvLSTM BODY BLOCK=================================
        self.pcd_align = PCD_Align(nf=nf, groups=groups, use_time=True)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=patch_size, input_dim=nf, hidden_dim=hidden_dim,
                                              kernel_size=(3, 3), num_layers=1, batch_first=True, front_RBs=front_RBs,
                                              groups=groups)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        layersAtBOffset = []
        layersAtBOffset.append(nn.Conv2d(128, 64, 3, 1, 1, bias=True))
        layersAtBOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersAtBOffset.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.layersAtBOffset = nn.Sequential(*layersAtBOffset)
        self.layersAtB = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8)

        layersCtBOffset = []
        layersCtBOffset.append(nn.Conv2d(128, 64, 3, 1, 1, bias=True))
        layersCtBOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersCtBOffset.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.layersCtBOffset = nn.Sequential(*layersCtBOffset)
        self.layersCtB = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8)

        layersFusion = []
        layersFusion.append(nn.Conv2d(192, 192, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(192, 192, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(192, 192, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(192, 64, 1, 1, 0, bias=True))
        self.layersFusion = nn.Sequential(*layersFusion)
        # ===============Reconstruction TAIL Block===========================
        # self.recon_trunk = mutil.make_layer(NeXtBlock_f, back_RBs)
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)

        # upsmapling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        """
        B, N, C, H, W or B, T, C, H, W
        Batch_Size, Frames, Channel, Height, Weight
        :param x:
        :return:
        """
        B, N, C, H, W = x.size()  # [6, 4, 3, 32, 32]

        original_input = x.view(-1, C, H, W)

        # Extract LR features
        LR_fea = self.lrelu(self.conv_first(original_input))  # -1 to let PT sort dim for B*N, [24, 3, 32, 32]
        LR_fea = self.feature_extraction(LR_fea)  # output size is [24, 64, 32, 32]
        LR_fea = LR_fea.view(B, N, -1, H, W)  # output size is [6, 4, 64, 32, 32]

        # Mixture of Expert System (Interpolation)
        # sliding windows
        to_lstm_fea = []


        for idx in range(N-1):
            fea1 = LR_fea[:, idx, :, :, :].clone()
            fea2 = LR_fea[:, idx+1, :, :, :].clone()
            if idx == 0:
                # gate from the original LR input for 1st interpolation
                gate_prob = self.gate1(original_input)
            elif idx == 1:
                # gate from the original LR input for 2nd interpolation
                gate_prob = self.gate2(original_input)
            else:
                # gate from the original LR input for 3rd interpolation
                gate_prob = self.gate3(original_input)
            dispatcher = SparseDispatcher(num_experts=4, gates=gate_prob, device=self.device)
            if idx == 0:
                to_lstm_fea.append(fea1)

            expert_inputs, non_zero = dispatcher.dispatch(LR_fea)
            gates = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            interpolated_fea = dispatcher.combine(expert_outputs)

            to_lstm_fea.append(interpolated_fea)
            to_lstm_fea.append(fea2)

        dnc_feats = torch.stack(to_lstm_fea, dim=1)
        print(dnc_feats.size())
        # B, T, C, H, W = interpolate_feats.size()
        print("can work until here")
        back_feats = dnc_feats

        B, T, C, H, W = dnc_feats.size()
        dnc_feats = dnc_feats.view(B, T, C, H, W)
        feats_non_linear_comparison = []
        for i in range(T):
            if i == 0:
                idx = [0, 0, 1]
            else:
                if i == T - 1:
                    idx = [T - 2, T - 1, T - 1]
                else:
                    idx = [i - 1, i, i + 1]
            fea0 = dnc_feats[:, idx[0], :, :, :].contiguous()
            fea1 = dnc_feats[:, idx[1], :, :, :].contiguous()
            fea2 = dnc_feats[:, idx[2], :, :, :].contiguous()
            AtBOffset = self.layersAtBOffset(torch.cat([fea0, fea1], dim=1))
            fea0_aligned = self.lrelu(self.layersAtB(fea0, AtBOffset))

            CtBOffset = self.layersCtBOffset(torch.cat([fea2, fea1], dim=1))
            fea2_aligned = self.lrelu(self.layersCtB(fea2, CtBOffset))

            feats_non_linear_comparison.append(self.layersFusion(torch.cat([fea0_aligned, fea1, fea2_aligned], dim=1)))
        feats_after_comparison = torch.stack(feats_non_linear_comparison, dim=1)
        lstm_feats = dnc_feats + feats_after_comparison.view(B, T, C, H, W)

        feats = self.ConvBLSTM(lstm_feats)

        feats = feats.view(B * T, C, H, W)
        # Reconstruction
        out = self.recon_trunk(feats)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        return outs