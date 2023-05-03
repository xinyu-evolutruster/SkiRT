import torch
import numpy as np
from pytorch3d.ops import knn_points, ball_query

from lib.losses import normal_loss, chamfer_loss_separate, repulsion_loss
from lib.utils_model import gen_transf_mtx_from_vtransf, get_transf_mtx_from_vtransf
from lib.utils_io import customized_export_ply

# fix the random seed
torch.manual_seed(12345)


def train_coarse(
    epoch, model, train_loader, optimizer, 
    shape_featmap,
    device='cuda',
    loss_weights=None,
):
    n_train_samples = len(train_loader.dataset)  

    train_m2s, train_s2m, train_lnormal, train_rgl, train_total = 0.0, 0.0, 0.0, 0.0, 0.0
    # w_s2m, w_m2s, w_normal, w_rgl = loss_weights
    w_s2m, w_m2s, w_normal, w_rgl = 1e4, 1e4, 1.0, 2e3

    model.train()
    for step, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        [query_points, associated_indices, body_verts, target_pc, target_pc_n, vtransf, index] = data
        query_points, vtransf = query_points.to(device), vtransf.to(device)
        body_verts, target_pc, target_pc_n = body_verts.to(device), target_pc.to(device), target_pc_n.to(device)
        
        # now we only train on a single garment so there is no need for indexing
        query_shape_code = shape_featmap

        # batch size
        bs = query_points.shape[0]
        query_points = query_points.float()
        pred_residuals, pred_normals = model(query_points, query_shape_code)

        # map the predicted points to the global coordinate system
        # --------------------------------
        # ------------ losses ------------
        
        indices = associated_indices.to(device)
 
        B, _ = indices.shape[:2]
        _, V = vtransf.shape[:2]
        
        # print("indices.shape: ", indices.shape)
        
        view_shape = list(indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        averaged_points = body_verts[batch_indices, indices, :]
        # averaged_points = torch.sum(averaged_points * weights.unsqueeze(-1), dim=2)
        averaged_points = torch.mean(averaged_points, dim=2)
        
        # print("averaged_points.shape: ", averaged_points.shape)
        
        body_verts = torch.cat([body_verts, averaged_points], dim=1)
        
        new_vtransf = vtransf.view(B, V, -1)
        new_vtransf = new_vtransf[batch_indices, indices, :]
        # averaged_vtransf = torch.sum(new_vtransf * weights.unsqueeze(-1), dim=2)
        averaged_vtransf = torch.mean(new_vtransf, dim=2)
        averaged_vtransf = averaged_vtransf.view(B, -1, 3, 3)
        transf_mtx_map = torch.cat([vtransf, averaged_vtransf], dim=1)
        
        pred_residuals = pred_residuals.unsqueeze(-1)
        pred_normals = pred_normals.unsqueeze(-1)

        pred_residuals = torch.matmul(transf_mtx_map, pred_residuals).squeeze(-1)
        pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
        
        # chamfer loss
        # Chamfer dist from the (s)can to (m)odel: from the GT points to its closest ponit in the predicted point set
        full_pred = (body_verts + pred_residuals).float()
        target_pc = target_pc.float()
        # full_pred = torch.randn(8, 10475, 3).to(device)
        
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

        # regularization loss
        rgl_loss = torch.mean(pred_residuals ** 2)
        
        # repulsion loss
        # rep_loss = repulsion_loss(full_pred)

        loss = s2m * w_s2m + m2s * w_m2s + lnormal * w_normal  + rgl_loss * w_rgl

        loss.backward()
        optimizer.step()
        
        # accumulate stats
        train_m2s += m2s * bs
        train_s2m += s2m * bs
        train_lnormal += lnormal * bs
        train_rgl += rgl_loss * bs
        train_total += loss * bs
        
        print("epoch: {}, step: {}, loss: {:.6f}, s2m: {:.6f}, m2s: {:.6f}, lnormal: {:.6f}, rgl: {:.6f}".format(
            epoch, step, loss, s2m, m2s, lnormal, rgl_loss
        ))
        
    train_m2s /= n_train_samples
    train_s2m /= n_train_samples
    train_lnormal /= n_train_samples
    train_rgl /= n_train_samples
    train_total /= n_train_samples
    
    return full_pred, averaged_points, pred_normals, query_points, train_m2s, train_s2m, train_lnormal, train_rgl, train_total
    

def train_fine(
    epoch, 
    coarse_model, fine_model, 
    train_loader, optimizer,
    global_shape_featmap, local_shape_featmap, 
    loss_weights, device='cuda'
):
    
    n_train_samples = len(train_loader.dataset)  

    train_m2s, train_s2m, train_lnormal, train_rdl, train_rgl, train_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # w_s2m, w_m2s, w_normal, w_rgl = loss_weights
    w_s2m, w_m2s, w_normal, w_rdl, w_rgl = 1e4, 1e4, 1e3, 2e3, 1.0
    
    # if epoch > 100:
    #     w_normal = 5.0
    
    # fix the parameters of the coarse model
    coarse_model.requires_grad_(False)
    global_shape_featmap.requires_grad_(False)
    # only train the fine model
    fine_model.train()
    local_shape_featmap.requires_grad_(True)
    
    for step, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        [query_points, indices, associated_indices, bary_coords, body_verts, target_pc, target_pc_n, vtransf, index] = data
        query_points, indices, vtransf = query_points.to(device), indices.to(device), vtransf.to(device)
        associated_indices, bary_coords = associated_indices.to(device), bary_coords.to(device)
        body_verts, target_pc, target_pc_n = body_verts.to(device), target_pc.to(device), target_pc_n.to(device)
            
        # now we only train on a single garment so there is no need for indexing
        global_shape_code = global_shape_featmap.to(device)
        local_shape_code = local_shape_featmap.to(device)

        # batch size
        bs = query_points.shape[0]
        query_points = query_points.float()
        
        B, _ = indices.shape[:2]
        _, V = vtransf.shape[:2]
        
        view_shape = list(associated_indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(associated_indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        averaged_points = body_verts[batch_indices, associated_indices, :]
        averaged_points = torch.mean(averaged_points, dim=2)
    
        new_vtransf = vtransf.view(B, V, -1)
        new_vtransf = new_vtransf[batch_indices, associated_indices, :]
        averaged_vtransf = torch.mean(new_vtransf, dim=2)
        averaged_vtransf = averaged_vtransf.view(B, -1, 3, 3)
        
        # sampling on the posed body model
        vtransf, body_verts = get_transf_mtx_from_vtransf(vtransf, body_verts, bary_coords, indices)
        transf_mtx_map = torch.cat([vtransf, averaged_vtransf], dim=1)
        body_verts = torch.cat([body_verts, averaged_points], dim=1)
        
        coarse_pred_residuals, coarse_pred_normals = coarse_model(
            query_points, global_shape_code
        )
        
        N = body_verts.shape[1]
        geom_feat = local_shape_code[:N, :]
        fine_pred_residuals, fine_pred_normals = fine_model(
            xyz=body_verts, 
            points=None,
            geom_feat=geom_feat
        )

        # combine the coarse and fine predictions
        pred_residuals = coarse_pred_residuals + fine_pred_residuals
        pred_normals = coarse_pred_normals + fine_pred_normals

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

        # regularization loss
        rdl_loss = torch.mean(pred_residuals ** 2)
        # rgl_loss = (local_shape_code ** 2).sum(dim=-1).mean()
        rgl_loss = (local_shape_code ** 2).sum()
        
        # repulsion loss
        # rep_loss = repulsion_loss(full_pred)

        loss = s2m * w_s2m + m2s * w_m2s + lnormal * w_normal  + rdl_loss * w_rdl + rgl_loss * w_rgl

        loss.backward()
        optimizer.step()
        
        # accumulate stats
        train_m2s += m2s * bs
        train_s2m += s2m * bs
        train_lnormal += lnormal * bs
        train_rdl += rdl_loss * bs
        train_rgl += rgl_loss * bs
        train_total += loss * bs
        
        print("epoch: {}, step: {}, loss: {:.6f}, s2m: {:.6f}, m2s: {:.6f}, lnormal: {:.6f}, rdl: {:.6f}, rgl: {:.6f}".format(
            epoch, step, loss, s2m, m2s, lnormal, rdl_loss, rgl_loss
        ))
        
    train_m2s /= n_train_samples
    train_s2m /= n_train_samples
    train_lnormal /= n_train_samples
    train_rdl /= n_train_samples
    train_rgl /= n_train_samples
    train_total /= n_train_samples
    
    return full_pred, body_verts, pred_normals, train_m2s, train_s2m, train_lnormal, train_rdl, train_rgl, train_total


def train_fine_upsample(
    epoch, 
    coarse_model, fine_model,
    train_loader, optimizer,
    global_shape_featmap, local_shape_featmap, 
    loss_weights, device='cuda'
):
    
    n_train_samples = len(train_loader.dataset)  

    train_m2s, train_s2m, train_lnormal, train_rdl, train_rgl, train_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # w_s2m, w_m2s, w_normal, w_rgl = loss_weights
    w_s2m, w_m2s, w_normal, w_rdl, w_rgl = 1e4, 1e4, 0.0, 2e3, 1.0
    
    # repulsion loss
    w_rep = 1e3
    train_rep = 0.0
    
    if epoch > 100:
        w_normal = 5.0
    
    # fix the parameters of the coarse model
    coarse_model.requires_grad_(False)
    global_shape_featmap.requires_grad_(False)
    # only train the fine model
    fine_model.train()
    local_shape_featmap.requires_grad_(True)
    
    for step, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        [query_points, indices, associated_indices, bary_coords, body_verts, target_pc, target_pc_n, vtransf, index] = data
        query_points, vtransf = query_points.to(device), vtransf.to(device)
        indices, bary_coords, associated_indices = indices.to(device), bary_coords.to(device), associated_indices.to(device)
        body_verts, target_pc, target_pc_n = body_verts.to(device), target_pc.to(device), target_pc_n.to(device)
            
        # now we only train on a single garment so there is no need for indexing
        global_shape_code = global_shape_featmap.to(device)
        local_shape_code = local_shape_featmap.to(device)

        # batch size
        bs = query_points.shape[0]
        query_points = query_points.float()
        
        B, V = vtransf.shape[:2]
        
        view_shape = list(associated_indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(associated_indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        averaged_points = body_verts[batch_indices, associated_indices, :]
        averaged_points = torch.mean(averaged_points, dim=2)
    
        new_vtransf = vtransf.view(B, V, -1)
        new_vtransf = new_vtransf[batch_indices, associated_indices, :]
        averaged_vtransf = torch.mean(new_vtransf, dim=2)
        averaged_vtransf = averaged_vtransf.view(B, -1, 3, 3)
        
        # sampling on the posed body model
        vtransf, body_verts = get_transf_mtx_from_vtransf(vtransf, body_verts, bary_coords, indices)
        # transf_mtx_map = torch.cat([vtransf, averaged_vtransf], dim=1)
        # body_verts = torch.cat([body_verts, averaged_points], dim=1)
        
        coarse_pred_residuals, coarse_pred_normals = coarse_model(
            query_points, global_shape_code
        )
       
        N = body_verts.shape[1] + averaged_points.shape[1]
        geom_feat = local_shape_code[:N, :]
        # get the upsampled predictions
        fine_pred_residuals, fine_pred_normals = fine_model(
            xyz=body_verts, 
            points=None,
            geom_feat=geom_feat,
            random_points=averaged_points
        )
        
        # tmp fixed
        upsampling_rate = 10
        # replicate the coarse pred residuals
        # coarse_pred_residuals = coarse_pred_residuals.unsqueeze(dim=1).repeat(1, upsampling_rate, 1, 1).view(B, -1, 3)
        # coarse_pred_normals = coarse_pred_normals.unsqueeze(dim=1).repeat(1, upsampling_rate, 1, 1).view(B, -1, 3)
        coarse_random_residuals = coarse_pred_residuals[:, 40000:, :].unsqueeze(dim=1).repeat(1, upsampling_rate, 1, 1).view(B, -1, 3)
        coarse_random_normals = coarse_pred_normals[:, 40000:, :].unsqueeze(dim=1).repeat(1, upsampling_rate, 1, 1).view(B, -1, 3)
        coarse_pred_residuals = torch.cat([coarse_pred_residuals[:, :40000, :], coarse_random_residuals], dim=1)
        coarse_pred_normals = torch.cat([coarse_pred_normals[:, :40000, :], coarse_random_normals], dim=1)
        # replicate body verts
        averaged_points = averaged_points.unsqueeze(dim=1).repeat(1, upsampling_rate, 1, 1).view(B, -1, 3)
        # replicate transf_mtx_map
        averaged_vtransf = averaged_vtransf.unsqueeze(dim=1).repeat(1, upsampling_rate, 1, 1, 1).view(B, -1, 3, 3)
        transf_mtx_map = torch.cat([vtransf, averaged_vtransf], dim=1)
        body_verts = torch.cat([body_verts, averaged_points], dim=1)

        # combine the coarse and fine predictions
        pred_residuals = coarse_pred_residuals + fine_pred_residuals
        pred_normals = coarse_pred_normals + fine_pred_normals

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

        # regularization loss
        rdl_loss = torch.mean(pred_residuals ** 2)
        # rgl_loss = (local_shape_code ** 2).sum(dim=-1).mean()
        rgl_loss = (local_shape_code ** 2).sum()
        
        # repulsion loss
        rep_loss = repulsion_loss(full_pred)
        # rep_loss = 0.0

        loss = s2m * w_s2m + m2s * w_m2s + lnormal * w_normal  + rdl_loss * w_rdl + rgl_loss * w_rgl + rep_loss * w_rep

        loss.backward()
        optimizer.step()
        
        # accumulate stats
        train_m2s += m2s * bs
        train_s2m += s2m * bs
        train_lnormal += lnormal * bs
        train_rdl += rdl_loss * bs
        train_rgl += rgl_loss * bs
        train_rep += rep_loss * bs
        train_total += loss * bs
        
        print("epoch: {}, step: {}, loss: {:.6f}, s2m: {:.6f}, m2s: {:.6f}, lnormal: {:.6f}, rdl: {:.6f}, rgl: {:.6f}, rep: {:.6f}".format(
            epoch, step, loss, s2m, m2s, lnormal, rdl_loss, rgl_loss, rep_loss
        ))
        
    train_m2s /= n_train_samples
    train_s2m /= n_train_samples
    train_lnormal /= n_train_samples
    train_rdl /= n_train_samples
    train_rgl /= n_train_samples
    train_rep /= n_train_samples
    train_total /= n_train_samples
    
    return full_pred, averaged_points, pred_normals, train_m2s, train_s2m, train_lnormal, train_rdl, train_rgl, train_rep, train_total