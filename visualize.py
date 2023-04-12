import os
import numpy as np
import open3d as o3d
import pytorch3d
import imageio

import torch

from vis.render_utils import render_turntable_pcl
from vis.utils import get_device


def render_single_pose():
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


def render_pose_sequence():
    # render a series of ...
    from lib.dataset import SKiRTCoarseDataset
    from lib.network import SkiRTCoarseNetwork
    from lib.utils_model import get_transf_mtx_from_vtransf
    from lib.utils_io import vertex_normal_2_vertex_color
    from torch.utils.data import DataLoader
    from vis.render_generic import render_point_cloud
    
    data_root = './data/resynth/packed'
    dataset = SKiRTCoarseDataset(
        dataset_type='resynth',
        data_root=data_root, 
        dataset_subset_portion=1.0,
        sample_spacing = 1,
        outfits={'rp_felice_posed_004'},
        smpl_model_path='./assets/smplx/smplx_model.obj',
        split='train', 
        num_samples=30000,
        body_model='smplx',
    )
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    model = SkiRTCoarseNetwork(
        input_nc=3,    # num channels of the input point
        input_sc=256,  # num channels of the shape code
        num_layers=5,   # num of the MLP layers
        num_layers_loc=3, # num of the MLP layers for the location prediction
        num_layers_norm=3, # num of the MLP layers for the normal prediction
        hsize=256,     # hidden layer size of the MLP
        skip_layer=[4],  # skip layers
        actv_fn='softplus',  # activation function
        pos_encoding=True
    )
    
    ckpt_dir = './checkpoints/SkiRT_coarse'
    ckpt_path = os.path.join(ckpt_dir, 'model_latest.pth')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model = model.cuda()
    
    featmap_path = os.path.join(ckpt_dir, 'geom_featmap_latest.pth')
    shape_featmap = torch.load(featmap_path)
    
    device = 'cuda'
    
    result_dir = './results/rp_felice_posed_004_vis'
    
    for step, data in enumerate(train_loader):
        [query_points, indices, bary_coords, body_verts, target_pc, target_pc_n, vtransf, index] = data
        query_points, indices, bary_coords, vtransf = query_points.to(device), indices.to(device), bary_coords.to(device), vtransf.to(device)
        body_verts, target_pc, target_pc_n = body_verts.to(device), target_pc.to(device), target_pc_n.to(device)
        
        query_shape_code = shape_featmap.to(device)
        
        bs = query_points.shape[0]
        query_points = query_points.float()
        pred_residuals, pred_normals = model(query_points, query_shape_code)

        transf_mtx_map, body_verts = get_transf_mtx_from_vtransf(vtransf, body_verts, bary_coords, indices)
        # print("transf_mtx_map shape: ", transf_mtx_map.shape) # [B, num_points, 3, 3]
        
        # map the predicted points to the global coordinate system
        pred_residuals = pred_residuals.unsqueeze(-1)
        pred_normals = pred_normals.unsqueeze(-1)

        transf_mtx_map = transf_mtx_map.float()
        pred_residuals = torch.matmul(transf_mtx_map, pred_residuals).squeeze(-1)
        pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

        print("pred_normals shape: ", pred_normals.shape)

        full_pred = (body_verts + pred_residuals).float().detach()
        rgb = vertex_normal_2_vertex_color(pred_normals[0].detach())
        rgb = torch.from_numpy(rgb.reshape(1, -1, 3)).float().cuda()
        
        rend = render_point_cloud(
            full_pred, rgb, image_size=512, background_color=(1, 1, 1), 
            point_radius=0.005, camera_dist=2, device=None
        )
        imageio.imwrite(os.path.join(result_dir, f'pred_pcd_{index[0]}.png'), rend)
        
        # render_point_cloud(
        #     point_cloud_path="data/bridge_pointcloud.npz",
        #     image_size=256,
        #     background_color=(1, 1, 1),
        #     device=None,
        # )


def main():
    
    render_pose_sequence()
    
    result_dir = './results/rp_felice_posed_004_vis'
    imgs = []
    cnt = 0
    for i in range(637):
        imgs.append(imageio.imread(os.path.join(result_dir, 'pred_pcd_{}.png'.format(i))))

    imageio.mimsave(os.path.join(result_dir, 'pred_pcd.gif'), imgs, fps=15)


if __name__ == '__main__':
    main()