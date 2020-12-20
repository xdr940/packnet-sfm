# copy rights xdr940

# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from cv2 import imwrite
from tqdm import tqdm
from path import Path

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor

#xdr940
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


@torch.no_grad()
def infer_and_save_depth(input_file, out_dir, model_wrapper, image_shape, half, save,dataset):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    basename=''
    if dataset=='kitti':
        input_file = Path(input_file)
        frame = input_file.stem
        sence = input_file.split('/')[-4]
        basename = sence+'_'+frame+'.{}'.format(save)



    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    #rgba 2 rgb
    image = image[:,:3,:,:]

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]



    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        out_dir=Path(out_dir)

        write_depth(out_dir/basename, depth=inv2depth(pred_inv_depth))


def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()


    #prepare input files
    files = split2files(data_path=args.data_path,split_txt=args.input)
    files.sort()
    print0('Found {} files'.format(len(files)))
    print('-> save at {}'.format(args.output))
    out_dir = Path(args.output)
    out_dir.mkdir_p()
    # Process each file
    for fn in tqdm(files):
        infer_and_save_depth(fn, out_dir, model_wrapper, image_shape, args.half, args.save,'kitti')


if __name__ == '__main__':
    args = parse_args()
    main(args)
