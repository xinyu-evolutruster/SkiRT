import os
from os.path import join, basename, dirname, realpath
import sys
import time
from datetime import date, datetime
import yaml

PROJECT_DIR = dirname(realpath(__file__))
LOGS_PATH = join(PROJECT_DIR, 'checkpoints')

SAMPLES_PATH = join(PROJECT_DIR, 'results', 'saved_samples')
sys.path.append(PROJECT_DIR)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import open3d as o3d

from lib.config_parser import parse_config, parse_outfits
from lib.dataset import SKiRTCoarseDataset, SKiRTFineDataset    
from lib.network import SkiRTCoarseNetwork, SkiRTFineNetwork, SkiRTUpsampleNetwork
from lib.train import train_coarse, train_fine, train_fine_upsample
from lib.utils_io import customized_export_ply, vertex_normal_2_vertex_color
from lib.utils_model import SampleSquarePoints
from lib.utils_train import adjust_loss_weights

torch.manual_seed(7)
np.random.seed(7)

DEVICE = torch.device('cuda')


def main_SkiRT():
    args = parse_config()
    
    exp_name = args.name
    coarse_exp_name = args.coarse_name
    
    # NOTE: when using your custom data, modify the following path to where the packed data is stored.
    data_root = join(PROJECT_DIR, 'data', '{}'.format(args.dataset_type.lower()), 'packed')

    log_dir = join(PROJECT_DIR,'tb_logs/{}/{}'.format(date.today().strftime('%m%d'), exp_name))
    ckpt_dir = join(LOGS_PATH, exp_name)
    coarse_ckpt_dir = join(LOGS_PATH, coarse_exp_name)
    result_dir = join(PROJECT_DIR, 'results', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # geometric feature tensor
    geom_featmap = torch.ones(1, 256).normal_(mean=0., std=0.01).cuda()
    geom_featmap.requires_grad = True

    # build_model
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
    model = model.cuda()
    # print(model)
    
    # local geometric feature tensor
    N = 40000
    local_geom_featmap = torch.ones(60000, 64).normal_(mean=0., std=0.01).cuda()
    local_geom_featmap.requires_grad = True
    
    if args.mode == 'train_fine':
        fine_model = SkiRTFineNetwork()
        fine_model = fine_model.cuda()
        # print(fine_model)
    if args.mode == 'train_fine_upsample':
        fine_model = SkiRTUpsampleNetwork(upsample_rate=10)
        fine_model = fine_model.cuda()
        # print(fine_model)
    
    if args.load_checkpoint == True:
        
        if args.mode.lower() == 'train':
            # load model weights
            model_path = os.path.join(ckpt_dir, 'model_latest.pth')
            print("model_path: ", model_path)
            print("loading model from checkpoint {}...".format(model_path))
            model.load_state_dict(torch.load(model_path))
            # load geometry feature map
            geom_featmap_path = os.path.join(ckpt_dir, 'geom_featmap_latest.pth')
            geom_featmap = torch.load(geom_featmap_path)
            geom_featmap.requires_grad_(True)
        if args.mode.lower() in ['train_fine', 'train_fine_upsample']:
            # load model weights
            coarse_model_path = os.path.join(coarse_ckpt_dir, 'model_latest.pth')
            print("coarse_model_path: ", coarse_model_path)
            print("loading coarse model from checkpoint {}...".format(coarse_model_path))
            model.load_state_dict(torch.load(coarse_model_path))
            
            fine_model_path = os.path.join(ckpt_dir, 'model_latest.pth')
            print("fine_model_path: ", fine_model_path)
            print("loading fine model from checkpoint {}...".format(fine_model_path))
            fine_model.load_state_dict(torch.load(fine_model_path))
            
            # load geometry feature map
            geom_featmap_path = os.path.join(coarse_ckpt_dir, 'geom_featmap_latest.pth')
            geom_featmap = torch.load(geom_featmap_path)
            geom_featmap.requires_grad_(False)
            
            local_geom_featmap_path = os.path.join(ckpt_dir, 'local_geom_featmap_latest.pth')
            local_geom_featmap = torch.load(local_geom_featmap_path)
            local_geom_featmap.requires_grad_(True)
    
    if args.mode.lower() in ['train', 'resume']:
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr},
                {"params": geom_featmap, "lr": args.lr_geomfeat}
            ])
    elif args.mode.lower() in ['train_fine', 'train_fine_upsample']:
        optimizer = torch.optim.Adam(
            [
                {"params": fine_model.parameters(), "lr": args.lr},
                {"params": local_geom_featmap, "lr": args.lr_geomfeat}
            ])
        
    if args.mode.lower() in ['train', 'resume']:

        train_set = SKiRTCoarseDataset(
            dataset_type='resynth',
            data_root=data_root, 
            dataset_subset_portion=1.0,
            sample_spacing = 1,
            outfits={args.outfit},
            smpl_model_path='./assets/smplx/SMPLX_NEUTRAL.npz',
            smpl_faces_path='./assets/smplx_faces.npy',
            split='train', 
            num_samples=N,
            body_model='smplx',
        )

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        print("Total: {} training examples. Training started..".format(len(train_set)))

        n_epochs = args.epochs
        epoch_now = 0

        start = time.time()
        pbar = range(epoch_now, n_epochs)

        writer = SummaryWriter(log_dir=log_dir)

        for epoch_idx in pbar:
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)
            loss_weights = torch.tensor([args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl, args.w_latent_rgl])

            train_stats = train_coarse(
                epoch_idx, model, train_loader, optimizer, shape_featmap=geom_featmap,
            )
            
            full_pred = train_stats[0]
            body_verts = train_stats[1]
            normals = train_stats[2]
            # query_points = train_stats[3]
            
            if epoch_idx % 10 == 0:
                customized_export_ply(
                    join(result_dir, 'pred_pcd_{}.ply'.format(epoch_idx)),
                    v=full_pred[0].cpu().detach().numpy(),
                    v_n=normals[0].cpu().detach().numpy(), 
                    v_c=vertex_normal_2_vertex_color(normals[0].cpu().detach().numpy()),                          
                )
                customized_export_ply(
                    join(result_dir, 'body_verts_{}.ply'.format(epoch_idx)),
                    v=body_verts[0].cpu().detach().numpy(),
                    v_n=normals[0][10475:].cpu().detach().numpy(), 
                    v_c=vertex_normal_2_vertex_color(normals[0][10475:].cpu().detach().numpy()), 
                )
                # customized_export_ply(
                #     join(result_dir, 'query_points_{}.ply'.format(epoch_idx)),
                #     v=query_points[0].cpu().detach().numpy(),
                #     v_n=normals[0].cpu().detach().numpy(), 
                #     v_c=vertex_normal_2_vertex_color(normals[0].cpu().detach().numpy()), 
                # )
            
            if epoch_idx % args.save_every == 0:
                pth = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch_idx))
                latest_pth = os.path.join(ckpt_dir, 'model_latest.pth')
                
                # save model
                torch.save(model.state_dict(), pth)
                torch.save(model.state_dict(), latest_pth)

                # save geometry feature map
                torch.save(geom_featmap, os.path.join(ckpt_dir, 'geom_featmap_{}.pth'.format(epoch_idx)))
                torch.save(geom_featmap, os.path.join(ckpt_dir, 'geom_featmap_latest.pth'))

            # train_m2s, train_s2m, train_lnormal, train_rgl, train_total
            tensorboard_tabs = ['model2scan', 'scan2model', 'normal_loss', 'residual_square', 'total_loss']
            stats = {'train': train_stats[4:]}

            for split in ['train']:
                for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                    writer.add_scalar('{}/{}'.format(tab, split), stat, epoch_idx)

            print("Epoch: {}, time: {:2f} minutes".format(epoch_idx, (time.time() - start) / 60))

        end = time.time()
        t_total = (end - start) / 60
        print("Training finished, duration: {:.2f} minutes. Now eval on test set..\n".format(t_total))

    elif args.mode.lower() == 'train_fine':

        train_set = SKiRTFineDataset(
            dataset_type='resynth',
            data_root=data_root, 
            dataset_subset_portion=1.0,
            sample_spacing = 1,
            outfits={args.outfit},
            smpl_model_path='./assets/smplx/SMPLX_NEUTRAL.npz',
            smpl_face_path='./assets/smplx_faces.npy',
            split='train', 
            num_samples=N,
            body_model='smplx',
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        print("Total: {} training examples. Training started..".format(len(train_set)))

        n_epochs = args.epochs
        epoch_now = 0

        model.to(DEVICE)
        start = time.time()
        pbar = range(epoch_now, n_epochs)

        writer = SummaryWriter(log_dir=log_dir)

        # load coarse checkpoint and fix the weights
        ckpt_path = os.path.join(coarse_ckpt_dir, 'model_latest.pth')
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        geom_featmap_path = os.path.join(coarse_ckpt_dir, 'geom_featmap_latest.pth')
        geom_featmap = torch.load(geom_featmap_path)
        geom_featmap.requires_grad_(False)

        for epoch_idx in pbar:
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)
            loss_weights = torch.tensor([args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl, args.w_latent_rgl])

            train_stats = train_fine(
                epoch_idx, 
                model, fine_model, 
                train_loader, optimizer,
                geom_featmap, local_geom_featmap, 
                loss_weights, device='cuda')
            
            full_pred = train_stats[0]
            # body_verts = train_stats[1]
            normals = train_stats[2]
            
            if epoch_idx % 10 == 0:
                customized_export_ply(
                    join(result_dir, 'pred_pcd_{}.ply'.format(epoch_idx)),
                    v=full_pred[0].cpu().detach().numpy(),
                    v_n=normals[0].cpu().detach().numpy(), 
                    v_c=vertex_normal_2_vertex_color(normals[0].cpu().detach().numpy()),                          
                )
            
            if epoch_idx % args.save_every == 0:
                pth = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch_idx))
                latest_pth = os.path.join(ckpt_dir, 'model_latest.pth')
                
                # save model
                torch.save(fine_model.state_dict(), pth)
                torch.save(fine_model.state_dict(), latest_pth)

                # save geometry feature map
                torch.save(local_geom_featmap, os.path.join(ckpt_dir, 'local_geom_featmap_{}.pth'.format(epoch_idx)))
                torch.save(local_geom_featmap, os.path.join(ckpt_dir, 'local_geom_featmap_latest.pth'))

            # train_m2s, train_s2m, train_lnormal, train_rgl, train_total
            tensorboard_tabs = ['model2scan', 'scan2model', 'normal_loss', 'residual_square', 'geomfeat_square', 'total_loss']
            stats = {'train': train_stats[3:]}

            for split in ['train']:
                for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                    writer.add_scalar('{}/{}'.format(tab, split), stat, epoch_idx)

            print("Epoch: {}, time: {}".format(epoch_idx, time.time() - start))
    
    elif args.mode.lower() == 'train_fine_upsample':
        
        train_set = SKiRTFineDataset(
            dataset_type='resynth',
            data_root=data_root, 
            dataset_subset_portion=1.0,
            sample_spacing = 1,
            outfits={'rp_beatrice_posed_025'},
            smpl_model_path='./assets/smplx/SMPLX_NEUTRAL.npz',
            smpl_face_path='./assets/smplx_faces.npy',
            split='train', 
            num_samples=N,
            body_model='smplx',
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        print("Total: {} training examples. Training started..".format(len(train_set)))

        n_epochs = args.epochs
        epoch_now = 0

        model.to(DEVICE)
        start = time.time()
        pbar = range(epoch_now, n_epochs)

        writer = SummaryWriter(log_dir=log_dir)

        # load coarse checkpoint and fix the weights
        ckpt_path = os.path.join(coarse_ckpt_dir, 'model_latest.pth')
        print("coarse_model_path: ", coarse_ckpt_dir)
        print("loading coarse model from checkpoint {}...".format(coarse_ckpt_dir))
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        
        geom_featmap_path = os.path.join(coarse_ckpt_dir, 'geom_featmap_latest.pth')
        geom_featmap = torch.load(geom_featmap_path)
        geom_featmap.requires_grad_(False)

        for epoch_idx in pbar:
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)
            loss_weights = torch.tensor([args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl, args.w_latent_rgl])

            train_stats = train_fine_upsample(
                epoch_idx, 
                model, fine_model, 
                train_loader, optimizer,
                geom_featmap, local_geom_featmap, 
                loss_weights, device='cuda')
            
            full_pred = train_stats[0]
            # body_verts = train_stats[1]
            normals = train_stats[2]
            
            if epoch_idx % 10 == 0:
                customized_export_ply(
                    join(result_dir, 'pred_pcd_{}.ply'.format(epoch_idx)),
                    v=full_pred[0].cpu().detach().numpy(),
                    v_n=normals[0].cpu().detach().numpy(), 
                    v_c=vertex_normal_2_vertex_color(normals[0].cpu().detach().numpy()),                          
                )

            if epoch_idx % args.save_every == 0:
                pth = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch_idx))
                latest_pth = os.path.join(ckpt_dir, 'model_latest.pth')
                
                # save model
                torch.save(fine_model.state_dict(), pth)
                torch.save(fine_model.state_dict(), latest_pth)

                # save geometry feature map
                torch.save(local_geom_featmap, os.path.join(ckpt_dir, 'local_geom_featmap_{}.pth'.format(epoch_idx)))
                torch.save(local_geom_featmap, os.path.join(ckpt_dir, 'local_geom_featmap_latest.pth'))

            # train_m2s, train_s2m, train_lnormal, train_rgl, train_total
            tensorboard_tabs = ['model2scan', 'scan2model', 'normal_loss', 'residual_square', 'geomfeat_square', 'rep_loss', 'total_loss']
            stats = {'train': train_stats[3:]}

            for split in ['train']:
                for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                    writer.add_scalar('{}/{}'.format(tab, split), stat, epoch_idx)

            print("Epoch: {}, time: {}".format(epoch_idx, time.time() - start))


        end = time.time()
        t_total = (end - start) / 60
        print("Training finished, duration: {:.2f} minutes. Now eval on test set..\n".format(t_total))



if __name__ == '__main__':
    # main()
    main_SkiRT()