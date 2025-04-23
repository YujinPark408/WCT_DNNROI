#!/usr/bin/env python

import argparse
import torch
import numpy as np

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='ts-model.ts',
                        metavar='FILE',
                        help="Specify the TorchScript file to convert"
                              " (default : 'ts-model.ts')")
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
    print('Parameters count: ', params)

if __name__ == "__main__":
    args = get_args()
    input_channels = args.input_channels
    output_channels = args.output_channels
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the TorchScript model
    print(f"Loading TorchScript model from {args.model}")
    ts_model = torch.jit.load(args.model, map_location=device)
    
    # Create a new instance of the specified network type
    if args.net_type == 'unet':
        net = UNet(input_channels, output_channels)
    elif args.net_type == 'uresnet':
        net = UResNet(input_channels, output_channels)
    elif args.net_type == 'nestedunet':
        net = NestedUNet(input_channels, output_channels)
    else:
        raise ValueError(f"Unsupported network type: {args.net_type}")
    
    net.to(device)
    
    # Extract state_dict from the TorchScript model
    ts_state_dict = {}
    for name, param in ts_model.named_parameters():
        ts_state_dict[name] = param.clone().detach()
    
    for name, buffer in ts_model.named_buffers():
        ts_state_dict[name] = buffer.clone().detach()
    
    # Try to load the state_dict into the PyTorch model
    try:
        net.load_state_dict(ts_state_dict)
        print("Successfully loaded parameters from TorchScript model")
        
        # Save the model's state_dict
        torch.save(net.state_dict(), args.output)
        print(f"Model state dictionary saved to {args.output}")
        
        # Verify the conversion
        example = torch.rand(1, input_channels, 512, 512, device=device)
        with torch.no_grad():
            ts_output = ts_model(example)
            net_output = net(example)
        
        if torch.allclose(ts_output, net_output, rtol=1e-3, atol=1e-3):
            print("✓ Verification successful: Models produce identical outputs")
        else:
            print("⚠ Warning: Models produce different outputs")
            print(f"  Max difference: {(ts_output - net_output).abs().max().item()}")
    
    except Exception as e:
        print(f"Failed to load parameters: {e}")
        print("Saving raw parameters instead")
        torch.save(ts_state_dict, args.output)
        print(f"Raw parameters saved to {args.output}")
