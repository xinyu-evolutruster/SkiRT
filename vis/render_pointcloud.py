import numpy as np

import torch
import pytorch3d
import imageio
import argparse
from matplotlib import pyplot as plt

from starter.render_generic import load_rgbd_data
from starter.utils import unproject_depth_image, get_device, get_points_renderer
from starter.render_utils import render_turntable_pcl


def render_pointcloud(
    pcl
):

    if device == None:
        device = get_device()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=int, default=3)
    args = parser.parse_args()

    render_pointcloud(args.num_frames, args.duration, args.image_size)
