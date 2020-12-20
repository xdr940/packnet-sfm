# copy rights xdr940

# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from cv2 import imwrite
from tqdm import tqdm
from path import Path
import cv2
import matplotlib.pyplot as plt

#
# from packnet_sfm.models.model_wrapper import ModelWrapper
# from packnet_sfm.datasets.augmentations import resize_image, to_tensor
# from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
# from packnet_sfm.utils.image import load_image
# from packnet_sfm.utils.config import parse_test_file
# from packnet_sfm.utils.load import set_debug
# from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
# from packnet_sfm.utils.logging import pcolor

#xdr940
from packnet_sfm.networks import packnet
from thrdparty.utils.fio import split2files

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt), model',
                        default='/home/roit/models/packnet/PackNet01_MR_selfsup_K.ckpt')

    parser.add_argument('--input', type=str, help='txt file',
                        default='/home/roit/datasets/splits/eigen_std/test_files.txt')
    parser.add_argument('--data_path',default='/home/roit/datasets/kitti')
    parser.add_argument('--datasets',default='mc')


    parser.add_argument('--output', type=str, help='Output file or folder',
                        default='/home/roit/bluep2/test_out/packnet/kitti_eigen')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default='png',
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args


def main(args):

    dataset='kitti'
    save = 'png'
    arch = packnet.PackNet01(version='1A', dropout=0.0)
    arch.eval()
    arch.to('cuda')

    loaded_dict = torch.load('/home/roit/models/packnet/PackNet01_MR_selfsup_K.pth', map_location='cuda')
    arch.load_state_dict(loaded_dict)




    #prepare input files
    files = split2files(data_path=args.data_path,split_txt=args.input)
    files.sort()
    print('-> Found {} files'.format(len(files)))
    print('-> save at {}'.format(args.output))
    out_dir = Path(args.output)
    out_dir.mkdir_p()
    # Process each file
    for fn in tqdm(files):
        img_np = cv2.imread(fn)
        img_np = cv2.resize(img_np, (640, 192))
        img_np = img_np.transpose([2, 0, 1])
        img = torch.tensor(img_np).to('cuda', dtype=torch.float32)
        img = img.unsqueeze(dim=0)

        disp = arch(img)
        disp = disp.detach().cpu().numpy()[0][0]

        if dataset == 'kitti':
            input_file = Path(fn)
            frame = input_file.stem
            sence = input_file.split('/')[-4]
            basename = sence + '_' + frame + '.{}'.format(save)

        plt.imsave(out_dir/basename,disp)



if __name__ == '__main__':
    args = parse_args()
    main(args)
