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
        num_frames=10,
        duration=3,
        image_size=256,
        device=None):

    if device == None:
        device = get_device()

    data = load_rgbd_data()
    rgb1 = torch.Tensor(data['rgb1'])
    rgb2 = torch.Tensor(data['rgb2'])
    mask1 = torch.Tensor(data['mask1'])
    mask2 = torch.Tensor(data['mask2'])
    depth1 = torch.Tensor(data['depth1'])
    depth2 = torch.Tensor(data['depth2'])
    camera1 = data['cameras1'].to(device)
    camera2 = data['cameras2'].to(device)

    points1, rgb1 = unproject_depth_image(rgb1, mask1, depth1, camera1)
    points2, rgb2 = unproject_depth_image(rgb2, mask2, depth2, camera2)

    pointcloud1 = pytorch3d.structures.Pointclouds(
        points=[points1], features=[rgb1])
    pointcloud2 = pytorch3d.structures.Pointclouds(
        points=[points2], features=[rgb2])

    # combine the two pointclouds
    points_combined = torch.cat((points1, points2), axis=0)
    rgb_combined = torch.cat((rgb1, rgb2), axis=0)
    pointcloud_combined = pytorch3d.structures.Pointclouds(
        points=[points_combined], features=[rgb_combined]
    )

    fps = num_frames // duration
    # the pointcloud corresponding to the first image
    renders1 = render_turntable_pcl(
        pointcloud1, image_size=image_size, num_frames=30, up=((0, -1, 0),), rotate_dir=False)
    imageio.mimsave("images/pcl1.gif", renders1,
                    fps=fps)

    # the pointcloud corresponding to the second image
    renders2 = render_turntable_pcl(
        pointcloud2, image_size=image_size, num_frames=30, up=((0, -1, 0),), rotate_dir=False)
    imageio.mimsave("images/pcl2.gif", renders2,
                    fps=fps)

    # the pointcloud formed by the union of the first 2 pointclouds
    renders_combined = render_turntable_pcl(
        pointcloud_combined, image_size=image_size, num_frames=30, up=((0, -1, 0),), rotate_dir=False
    )
    imageio.mimsave("images/pcl_combined.gif", renders_combined,
                    fps=fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=int, default=3)
    args = parser.parse_args()

    render_pointcloud(args.num_frames, args.duration, args.image_size)
