#!/usr/bin/env python

import argparse
import os
import numpy as np

import torch
import torchvision

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                              " (default : 'MODEL.pth')")
    parser.add_argument('--output', '-o', default='model-converted.pth',
                        metavar='FILE',
                        help="Output file path for the .pth model"
                              " (default : 'model-converted.pth')")
    parser.add_argument('--net-type', '-t', default='unet',
                        choices=['unet', 'uresnet', 'nestedunet'],
                        help="Network architecture (default: unet)")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use cuda version of the net",
                        default=False)
    parser.add_argument('--input-channels', '-i', type=int, default=3,
                        help="Number of input channels (default: 3)")
    parser.add_argument('--output-channels', '-c', type=int, default=1,
                        help="Number of output channels (default: 1)")

    return parser.parse_args()
def count_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params = ', params)

if __name__ == "__main__":
    args = get_args()
    input_channels = args.input_channels
    output_channels = args.output_channels

    if args.net_type == 'unet':
        net = UNet(input_channels, output_channels)
    elif args.net_type == 'uresnet':
        net = UResNet(input_channels, output_channels)
    elif args.net_type == 'nestedunet':
        net = NestedUNet(input_channels, output_channels)
    else:
        raise ValueError(f"Unsupported network type: {args.net_type}")

    count_params(net)

    example = torch.rand(1, input_channels, 800, 600)
    
    if args.gpu:
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        sm = torch.jit.trace(net, example.cuda())
        output = net(example.cuda())
        # print(output[0][0][0])
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        sm = torch.jit.trace(net, example)
        output = net(example)
        # print(output[0][0][0])

    sm.save('ts-model.ts')