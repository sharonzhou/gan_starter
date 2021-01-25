"""
base_arg_parser.py
Base arguments for all scripts
"""

import argparse
import json
import numpy as np
import os
import random
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='disentanglement')

        self.parser.add_argument('--name', type=str, default='debug', help='Experiment name prefix.')
        
        self.parser.add_argument('--model', type=str, choices=('dcgan', 'weak_disentangle'), default='weak_disentangle', help='Model to use')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size.') # 2048 fits
        self.parser.add_argument('--viz_batch_size', type=int, default=8, help='Visualization image batch size.')
        
        self.parser.add_argument('--dataset_name', type=str, default='dsprites', choices=('dpsrites', 'shapes3d', 'norb', 'cars3d', 'mpi3d', 'scream'), help='Dataset to use.')
        
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for the DataLoader.')
        
        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'), help='Initialization method to use for conv kernels and linear weights.')
        
        self.parser.add_argument('--save_dir', type=str, default='./results', help='Directory for results, prefix.')

        self.parser.add_argument('--higher_metric_better', type=bool, default=False, help='For evaluation, higher the metric the better, else lower.')

    def parse_args(self):
        args = self.parser.parse_args()

        # Create save dir for run
        args.name = args.name + '_' + datetime.now().strftime('%b%d_%H%M%S')
        save_dir = os.path.join(args.save_dir, f'{args.name}')
        os.makedirs(save_dir, exist_ok=False)
        args.save_dir = save_dir
        
        # Save args to a JSON file
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        # Create ckpt dir and viz dir
        args.ckpt_dir = os.path.join(args.save_dir, 'ckpts')
        os.makedirs(args.ckpt_dir, exist_ok=False)
        
        args.viz_dir = os.path.join(args.save_dir, 'viz')
        os.makedirs(args.viz_dir, exist_ok=False)

        # Set up available GPUs
        def args_to_list(csv, arg_type=int):
            """Convert comma-separated arguments to a list."""
            arg_vals = [arg_type(d) for d in str(csv).split(',')]
            return arg_vals

        args.gpu_ids = args_to_list(args.gpu_ids)

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
        else:
            args.device = 'cpu'
       
        if hasattr(args, 'supervised_factors'):
            args.supervised_factors = args_to_list(args.supervised_factors)


        if args.higher_metric_better:
            args.best_ckpt_metric = float('-inf')
        else:
            args.best_ckpt_metric = float('inf')
        return args
