# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()

        '''Commented out for custom backbone'''
        # for name, parameter in backbone.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}

        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):

        xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):

        ''' Commented out for custom backbone '''

        # backbone = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

        num_channels = 512
        super().__init__(self, train_backbone, num_channels, return_interm_layers = False)

class Custom_Backbone(nn.Module):
    ''' OUR OWN CUSTOM BACKBONE'''
    def __init__(self, name:str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        

        super(Custom_Backbone, self).__init__()

        # Number of data points per channel = 128 * 600

        self.num_channels = 10
        C = self.num_channels

        self.conv_mix = nn.Conv2d(1, C, kernel_size = (C,1))
        self.relu = nn.ReLU()

        self.conv_layer1 = nn.Conv2d(C, 2*C,kernel_size=(1,3), stride=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(2*C)
        
        self.conv_layer2 = nn.Conv2d(2*C, 4*C,kernel_size=(1,3), stride=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(4*C)

        self.conv_layer3 = nn.Conv2d(4*C, 8*C,kernel_size=(1,3), stride=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(8*C)

        self.conv_layer4 = nn.Conv2d(8*C, 16*C,kernel_size=(1,3), stride=(1,1))
        self.batch_norm4 = nn.BatchNorm2d(16*C)
    
        self.conv_layer5 = nn.Conv2d(16*C, 32*C,kernel_size=(1,3), stride=(1,1))
        self.batch_norm5 = nn.BatchNorm2d(32*C)

        BackboneBase(self, train_backbone, self.num_channels, return_interm_layers)


    def forward(self, tensor_list: NestedTensor):

        out = tensor_list.tensors
        out = self.conv_mix(out)
        out = self.relu(out)

        out = self.conv_layer1(out)
        out = self.batch_norm1(out)
        out = self.conv_layer2(out)
        out = self.batch_norm2(out)
        out = self.conv_layer3(out)
        out = self.batch_norm3(out)
        out = self.conv_layer4(out)
        out = self.batch_norm4(out)
        out = self.conv_layer5(out)
        out = self.batch_norm5(out)

        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks

    '''Use own backbone'''
    backbone = Custom_Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    #backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
