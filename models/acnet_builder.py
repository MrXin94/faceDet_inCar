# from builder import ConvBuilder
import torch.nn as nn
import numpy as np

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros", deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels,
                                            eps=0.001,  # value found in tensorflow
                                            momentum=0.1,  # default pytorch value
                                            affine=True
                                            )

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + int(np.sqrt(kernel_size)), center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + int(np.sqrt(kernel_size)))
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      )

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels,
                                         eps=0.001,  # value found in tensorflow
                                         momentum=0.1,  # default pytorch value
                                         affine=True
                                         )
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels,
                                         eps=0.001,  # value found in tensorflow
                                         momentum=0.1,  # default pytorch value
                                         affine=True
                                         )



    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs

class ACBlock_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode="zeros", deploy=False):
        super(ACBlock_relu, self).__init__()
        self.ACBlock = ACBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                 padding_mode=padding_mode, deploy=deploy)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ACBlock(x)
        x = self.relu(x)
        return x

# class ACNetBuilder(ConvBuilder):
#
#     def __init__(self, base_config, deploy):
#         super(ACNetBuilder, self).__init__(base_config=base_config)
#         self.deploy = deploy
#
#     def switch_to_deploy(self):
#         self.deploy = True
#
#
#     def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
#         if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
#             return super(ACNetBuilder, self).Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                  padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, use_original_conv=True)
#         else:
#             return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                        padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)
#
#
#     def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
#         if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
#             return super(ACNetBuilder, self).Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                  padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
#         else:
#             return ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                        padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)
#
#
#     def Conv2dBNReLU(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
#         if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
#             return super(ACNetBuilder, self).Conv2dBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                  padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
#         else:
#             se = nn.Sequential()
#             se.add_module('acb', ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                        padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy))
#             se.add_module('relu', self.ReLU())
#             return se
#
#
#     def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
#         if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
#             return super(ACNetBuilder, self).BNReLUConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                  padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=True)
#         bn_layer = self.BatchNorm2d(num_features=in_channels)
#         conv_layer = ACBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                        padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy)
#         se = self.Sequential()
#         se.add_module('bn', bn_layer)
#         se.add_module('relu', self.ReLU())
#         se.add_module('acb', conv_layer)
#         return se