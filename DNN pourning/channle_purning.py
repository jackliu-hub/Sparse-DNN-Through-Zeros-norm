import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import math
from copy import deepcopy


def to_var(x, requires_grad=False):
    """
    Automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)  # 定义权重乘以一个mask矩阵
        self.weight.data = self.weight.data * self.mask.data  # 乘以掩码矩阵
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.conv2d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)



"""Reference https://github.com/zepx/pytorch-weight-prune/"""
# ---------------------
def prune_rate(model, verbose=False):
    """
    Print out prune rate for each layer and the whole network

    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0
    # 对模性参数进行操作
    for parameter in model.parameters():
        # 参数层
        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim  # 参数个数
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy() == 0)  # 提取参数零元个数
            nb_zero_param += zero_param_this_layer  # 此层零元个数

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                    layer_id,
                    'Conv' if len(parameter.data.size()) == 4 \
                        else 'Linear',
                    100. * zero_param_this_layer / param_this_layer,  # 输出剪枝个数
                ))
    pruning_perc = 100. * nb_zero_param / total_nb_param  # 剪枝比例
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
        if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of
    kernel weights
    arXiv:1611.06440

    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():
        if len(p.data.size()) == 4:  # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1) \
                                   .sum(axis=1) / (p_np.shape[1] * p_np.shape[2] * p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                               np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)  # 将list 数据转换成 array

    # set mask corresponding to the filter to prune  #创建掩码使相应的卷积核被剪枝掉
    to_prune_layer_ind = np.argmin(values[:, 0])  # 找出要剪枝的层的索引
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])  # 要剪枝的核
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.  # 创建剪枝的掩码

    #     print('Prune filter #{} in layer #{}'.format(
    #         to_prune_filter_ind,
    #         to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc

    (not iterative pruning)
    '''
    masks = []  # 掩码记录
    current_pruning_perc = 0.  # 当前剪枝度

    while current_pruning_perc < pruning_perc:  # 此处设置为50%  出现问题
        masks = prune_one_filter(model, masks)  # 此处有问题   创建掩码
        model.set_masks(masks)  # 此处 有问题  #对此掩码与权重相乘   出现0除0 情况
        current_pruning_perc = prune_rate(model, verbose=False)  # 计算当前剪枝率
    #         print('{:.2f} pruned'.format(current_pruning_perc))
    return masks
