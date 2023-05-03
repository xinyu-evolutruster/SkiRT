import os
import numpy as np
import open3d as o3d
import pytorch3d
import imageio
from pytorch3d.ops import knn_points, ball_query

import torch

from vis.render_utils import render_turntable_pcl
from vis.utils import get_device
from lib.utils_io import customized_export_ply
from lib.losses import normal_loss, chamfer_loss_separate, repulsion_loss

# fix the random seed
torch.manual_seed(7)
np.random.seed(7)

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


# train_beatrice: 0.55, 0.86
# test_beatrice: 0.65, 0.92

# train_anna: 0.32, 0.85
# test_anna: 0.79, 0.92

# train_christine: 1.20, 1.12
# test_christine: 1.97, 1.20

# train_felice: 3.78, 1.96
# test_felice: 7.65, 2.13

def render_pose_sequence(result_dir):
    # render a series of ...
    from lib.dataset import SKiRTCoarseDataset, SKiRTFineDataset
    from lib.network import SkiRTCoarseNetwork, SkiRTFineNetwork, SkiRTUpsampleNetwork
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
        outfits={'rp_beatrice_posed_025'},
        smpl_model_path='./assets/smplx/SMPLX_NEUTRAL.npz',
        smpl_faces_path='./assets/smplx_faces.npy',
        split='train', 
        num_samples=10000,
        body_model='smplx',
    )
    # dataset = SKiRTFineDataset(
    #     dataset_type='resynth',
    #     data_root=data_root, 
    #     dataset_subset_portion=1.0,
    #     sample_spacing = 1,
    #     outfits={'rp_anna_posed_001'},
    #     smpl_face_path='./assets/smplx_faces.npy',
    #     smpl_model_path='./assets/smplx/SMPLX_NEUTRAL.npz',
    #     split='train', 
    #     num_samples=40000,
    #     body_model='smplx',
    # )
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, drop_last=True)
    
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
    fine_model = SkiRTFineNetwork()
    
    # print(fine_model.state_dict().keys())
    
    ckpt_dir = './checkpoints/SkiRT_coarse_rp_beatrice_025'
    ckpt_path = os.path.join(ckpt_dir, 'model_latest.pth')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model = model.cuda()
    
    featmap_path = os.path.join(ckpt_dir, 'geom_featmap_latest.pth')
    shape_featmap = torch.load(featmap_path)
    shape_featmap.requires_grad_(False)
    
    fine_ckpt_dir = './checkpoints/SkiRT_fine_rp_anna_001'
    fine_ckpt_path = os.path.join(fine_ckpt_dir, 'model_latest.pth')
    fine_model.load_state_dict(torch.load(fine_ckpt_path))
    # fine_model.eval()
    fine_model = fine_model.cuda()
    
    local_featmap_path = os.path.join(fine_ckpt_dir, 'local_geom_featmap_latest.pth')
    local_shape_featmap = torch.load(local_featmap_path)
    local_shape_featmap.requires_grad_(False)
    
    device = 'cuda'
    
    os.makedirs(result_dir, exist_ok=True)
    
    query_shape_code = shape_featmap.to(device)
    local_query_shape_code = local_shape_featmap.to(device)

    total_chamfer_loss = 0.0
    total_normal_loss = 0.0
    n_test_samples = len(train_loader.dataset)

    with torch.no_grad():
        for step, data in enumerate(train_loader):
            # [query_points, indices, associated_indices, bary_coords, body_verts, target_pc, target_pc_n, vtransf, index] = data
            [query_points, associated_indices, body_verts, target_pc, target_pc_n, vtransf, index] = data
            
            # query_points, indices, bary_coords, vtransf = query_points.to(device), indices.to(device), bary_coords.to(device), vtransf.to(device)
            query_points, vtransf = query_points.to(device), vtransf.to(device)
            associated_indices = associated_indices.to(device)
            # indices, associated_indices, bary_coords = indices.to(device), associated_indices.to(device), bary_coords.to(device)
            body_verts, target_pc, target_pc_n = body_verts.to(device), target_pc.to(device), target_pc_n.to(device)
            
            bs = query_points.shape[0]
            query_points = query_points.float()
            
            B, _ = associated_indices.shape[:2]
            _, V = vtransf.shape[:2]
            view_shape = list(associated_indices.shape)
            view_shape[1:] = [1] * (len(view_shape) - 1)
            repeat_shape = list(associated_indices.shape)
            repeat_shape[0] = 1
            batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
            averaged_points = body_verts[batch_indices, associated_indices, :]
            averaged_points = averaged_points.mean(dim=2) # [batch, num_points, 3]

            new_vtransf = vtransf.view(B, V, -1)
            new_vtransf = new_vtransf[batch_indices, associated_indices, :]
            averaged_vtransf = torch.mean(new_vtransf, dim=2)
            averaged_vtransf = averaged_vtransf.view(B, -1, 3, 3)
            
            # transf_mtx_map, body_verts = get_transf_mtx_from_vtransf(vtransf, body_verts, bary_coords, indices)
            transf_mtx_map = vtransf
            body_verts = torch.cat([body_verts, averaged_points], dim=1)
            transf_mtx_map = torch.cat([transf_mtx_map, averaged_vtransf], dim=1)
            
            coarse_pred_residuals, coarse_pred_normals = model(query_points, query_shape_code)
            
            N = body_verts.shape[1]
            # geom_feat = local_query_shape_code[:N, :]
            # fine_pred_residuals, fine_pred_normals = fine_model(
            #     xyz=body_verts, 
            #     points=None,
            #     geom_feat=geom_feat
            # )
        
            pred_residuals = coarse_pred_residuals #+ fine_pred_residuals
            pred_normals = coarse_pred_normals #+ fine_pred_normals

            # map the predicted points to the global coordinate system
            pred_residuals = pred_residuals.unsqueeze(-1)
            pred_normals = pred_normals.unsqueeze(-1)

            transf_mtx_map = transf_mtx_map.float()
            pred_residuals = torch.matmul(transf_mtx_map, pred_residuals).squeeze(-1)
            pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

            # --------------------------------
            # ------------ losses ------------
            
            # chamfer loss
            # Chamfer dist from the (s)can to (m)odel: from the GT points to its closest ponit in the predicted point set
            full_pred = (body_verts + pred_residuals).float()
            
            m2s, s2m, idx_closest_gt, _ = chamfer_loss_separate(full_pred, target_pc) #idx1: [#pred points]
            s2m = torch.mean(s2m)
            
            # normal loss
            lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt)
            
            # dist from the predicted points to their respective closest point on the GT, projected by
            # the normal of these GT points, to appxoimate the point-to-surface distance
            nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
            target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
            pc_diff = target_points_chosen - full_pred # vectors from prediction to its closest point in gt pcl
            m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
            m2s = torch.mean(m2s**2) # the length (squared) is the approx. pred point to scan surface dist.

            print("chamfer loss: {:.6f}, normal loss: {:.6f}".format(s2m+m2s, lnormal))
            total_chamfer_loss += (s2m.detach() + m2s.detach()) * 8
            total_normal_loss += lnormal.detach() * 8

            # full_pred = (posed_sampled_points + pred_residuals).float().detach()
            full_pred = (body_verts + pred_residuals).float().detach()
            
            print("step = ", step)
            for i in range(8):
                rgb = vertex_normal_2_vertex_color(pred_normals[i].detach())
                rgb = torch.from_numpy(rgb.reshape(1, -1, 3)).float().cuda()
                rend = render_point_cloud(
                    full_pred[i].unsqueeze(0), rgb, image_size=512, background_color=(255, 255, 255), 
                    point_radius=0.005, camera_dist=2.2, device=None
                )
                imageio.imwrite(os.path.join(result_dir, f'pred_pcd_{index[i]}.png'), rend)
            
            # if step % 5 == 0:
            #     #save the predicted point cloud
            #     customized_export_ply(
            #         os.path.join(result_dir, 'pred_pcd_{}.ply'.format(step)),
            #         v=full_pred[0].cpu().detach().numpy(),
            #         v_n=pred_normals[0].cpu().detach().numpy(), 
            #         v_c=vertex_normal_2_vertex_color(pred_normals[0].cpu().detach().numpy()),                          
            #     )
    
    total_chamfer_loss /= n_test_samples
    total_normal_loss /= n_test_samples
    print("total chamfer loss: {:.6f}, total normal loss: {:.6f}".format(total_chamfer_loss, total_normal_loss))

def main():
    
    result_dir = './results/rp_beatrice_posed_025_vis'
    render_pose_sequence(result_dir)
    
    imgs = []
    cnt = 0
    for i in range(630):
        imgs.append(imageio.imread(os.path.join(result_dir, 'pred_pcd_{}.png'.format(i))))

    imageio.mimsave(os.path.join(result_dir, 'pred_pcd.gif'), imgs, fps=15)


if __name__ == '__main__':
    main()