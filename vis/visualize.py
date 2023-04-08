import numpy as np
import open3d as o3d
import pytorch3d
import imageio

import torch

from render_utils import render_turntable_pcl
from utils import get_device


def main():
    pcl_path_anna = '../results/rp_anna_posed_001_normal_geomfeat/pred_pcd_250.ply'
    pcl_path_felice = '../results/rp_felice_posed_004/pred_pcd_220.ply'
    
    pcl_anna = o3d.io.read_point_cloud(pcl_path_anna)
    pcl_felice = o3d.io.read_point_cloud(pcl_path_felice)
    
    device = get_device()
    
    points_anna = torch.from_numpy(np.asarray(pcl_anna.points)).float().to(device)
    color_anna = torch.from_numpy(np.asarray(pcl_anna.normals)).float().to(device)
    
    points_felice = torch.from_numpy(np.asarray(pcl_felice.points)).float().to(device)
    color_felice = torch.from_numpy(np.asarray(pcl_felice.colors)).float().to(device)
    
    pointcloud_anna = pytorch3d.structures.Pointclouds(
        points=[points_anna], features=[color_anna]
    )
    pointcloud_felice = pytorch3d.structures.Pointclouds(
        points=[points_felice], features=[color_felice]
    )
    
    renders = render_turntable_pcl(
        pointcloud_anna, 45, 512, dist=2.5
    )
    imageio.mimsave("./pcl_anna.gif", renders, fps=15)

    renders = render_turntable_pcl(
        pointcloud_felice, 45, 512, dist=2.5
    )
    imageio.mimsave("./pcl_felice.gif", renders, fps=15)


if __name__ == '__main__':
    main()