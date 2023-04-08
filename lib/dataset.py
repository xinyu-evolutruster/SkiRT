import os
from os.path import join, dirname
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

import open3d as o3d
import pickle

from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class CloDataSet(Dataset):
    def __init__(self, data_root=None, split='train', body_model='smpl', dataset_type='cape',
                 sample_spacing=1, query_posmap_size=256, inp_posmap_size=128, scan_npoints=-1, 
                 dataset_subset_portion=1.0, outfits={}):

        self.dataset_type = dataset_type
        self.data_root = data_root
        self.data_dirs = {outfit: join(data_root, outfit, split) for outfit in outfits.keys()} # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.dataset_subset_portion = dataset_subset_portion # randomly subsample a number of data from each clothing type (using all data from all outfits will be too much)
        self.query_posmap_size = query_posmap_size
        self.inp_posmap_size = inp_posmap_size

        self.split = split
        self.query_posmap_size = query_posmap_size
        self.spacing = sample_spacing
        self.scan_npoints = scan_npoints
        self.f = np.load(join(SCRIPT_DIR, '..', 'assets', '{}_faces.npy'.format(body_model)))
        self.clo_label_def = outfits


        self.posmap, self.posmap_meanshape, self.scan_n, self.scan_pc = [], [], [], []
        self.scan_name, self.body_verts, self.clo_labels =  [], [], []
        self.vtransf = []
        self._init_dataset()
        self.data_size = int(len(self.posmap))

        print('Data loaded, in total {} {} examples.\n'.format(self.data_size, self.split))

    def _init_dataset(self):
        print('Loading {} data...'.format(self.split))

        flist_all = []
        subj_id_all = []

        for outfit_id, (outfit, outfit_datadir) in enumerate(self.data_dirs.items()):
            flist = sorted(glob.glob(join(outfit_datadir, '*.npz')))[::self.spacing]
            print('Loading {}, {} examples..'.format(outfit, len(flist)))
            flist_all = flist_all + flist
            subj_id_all = subj_id_all + [outfit.split('_')[0]] * len(flist)

        if self.dataset_subset_portion < 1.0:
            import random
            random.shuffle(flist_all)
            num_total = len(flist_all)
            num_chosen = int(self.dataset_subset_portion*num_total)
            flist_all = flist_all[:num_chosen]
            print('Total examples: {}, now only randomly sample {} from them...'.format(num_total, num_chosen))

        for idx, fn in enumerate(tqdm(flist_all)):
            dd = np.load(fn)
            clo_type = dirname(fn).split('/')[-2] # e.g. longlong 
            clo_label = self.clo_label_def[clo_type] # the numerical label of the type in the lookup table (outfit_labels.json)
            self.clo_labels.append(torch.tensor(clo_label).long())
            self.posmap.append(torch.tensor(dd['posmap{}'.format(self.query_posmap_size)]).float().permute([2,0,1]))

            # for historical reasons in the packed data the key is called "posmap_canonical"
            # it actually stands for the positional map of the *posed, mean body shape* of SMPL/SMPLX (see POP paper Sec 3.2)
            # which corresponds to the inp_posmap_ms in the train and inference code 
            # if the key is not available, simply use each subject's personalized body shape.
            if 'posmap_canonical{}'.format(self.inp_posmap_size) not in dd.files:
                self.posmap_meanshape.append(torch.tensor(dd['posmap{}'.format(self.inp_posmap_size)]).float().permute([2,0,1]))
            else:
                self.posmap_meanshape.append(torch.tensor(dd['posmap_canonical{}'.format(self.inp_posmap_size)]).float().permute([2,0,1]))
            self.scan_n.append(torch.tensor(dd['scan_n']).float())
            
            # in the packed files the 'scan_name' field doensn't contain subj id, need to append it
            scan_name_loaded = str(dd['scan_name'])
            scan_name = scan_name_loaded if scan_name_loaded.startswith('0') else '{}_{}'.format(subj_id_all[idx], scan_name_loaded)
            self.scan_name.append(scan_name)

            self.body_verts.append(torch.tensor(dd['body_verts']).float())
            self.scan_pc.append(torch.tensor(dd['scan_pc']).float())
            
            vtransf = torch.tensor(dd['vtransf']).float()
            if vtransf.shape[-1] == 4:
                vtransf = vtransf[:, :3, :3]
            self.vtransf.append(vtransf)


    def __getitem__(self, index):
        posmap = self.posmap[index]
        posmap_meanshape = self.posmap_meanshape[index] # in mean SMPL/ SMPLX body shape but in the same pose as the original subject
        scan_name = self.scan_name[index]
        body_verts = self.body_verts[index]
        clo_label = self.clo_labels[index]

        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]

        vtransf = self.vtransf[index]

        if self.scan_npoints != -1: 
            selected_idx = torch.randperm(len(scan_n))[:self.scan_npoints]
            scan_pc = scan_pc[selected_idx, :]
            scan_n = scan_n[selected_idx, :]

        return posmap, posmap_meanshape, scan_n, scan_pc, vtransf, scan_name, body_verts, clo_label

    def __len__(self):
        return self.data_size
    

class SKiRTCoarseDataset(Dataset):
    def __init__(
        self, 
        dataset_type='resynth',
        data_root=None, 
        dataset_subset_portion=1.0,
        sample_spacing = 1,
        outfits={},
        smpl_face_path=None,
        smpl_model_path=None,
        split='train', 
        num_samples=20000,
        knn=3,
        body_model='smplx'):
        
        self.dataset_type = dataset_type
        self.data_root = data_root
        self.split = split
        self.spacing = sample_spacing
        
        # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.data_dirs = {outfit: join(data_root, outfit, split) for outfit in outfits} 
        
        # randomly subsample a number of data from each clothing type (using all data from all outfits will be too much)
        self.dataset_subset_portion = dataset_subset_portion 
        
        # import smplx model
        self.smpl_sampled_points = None
        self.faces, self.vertices = None, None
        if body_model == 'smplx':
            self.faces = np.load(smpl_face_path, allow_pickle=True)
            # the input file format is .pkl
            self.vertices = pickle.load(open(smpl_model_path, 'rb'), encoding='latin1')['v_template']
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self.vertices),
                o3d.utility.Vector3iVector(self.faces)
            )
            # sample points from the model
            pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
            
            self.smpl_sampled_points = np.asarray(pcd.points)
        else:
            raise NotImplementedError('Only support SMPLX model for now.')
        
        # find the LBS (Linear Blending Skinning) weights for each sampled points
        tree_smpl_verts = KDTree(self.vertices)
        # find the closest 3 vertices for each sampled point
        distances, indices = tree_smpl_verts.query(self.smpl_sampled_points, k=knn)
        
        total_distances = distances.sum(axis=1)
        barycentric_coords = distances / total_distances[:, None]
        triangles = self.vertices[indices]
        
        self.smpl_sampled_points = barycentric_coords[:, 0, None] * triangles[:, 0, :] + barycentric_coords[:, 1, None] * triangles[:, 1, :] + barycentric_coords[:, 2, None] * triangles[:, 2, :]
        self.barycentric_coords = barycentric_coords
        self.indices = indices
    
        # load the data. Train with only one garment template first.
        flist_all = []
        subj_id_all = []
        
        for outfit_id, (outfit, outfit_datadir) in enumerate(self.data_dirs.items()):
            flist = sorted(glob.glob(join(outfit_datadir, '*.npz')))[::self.spacing]
            print('Loading {}, {} examples..'.format(outfit, len(flist)))
            flist_all = flist_all + flist
            subj_id_all = subj_id_all + [outfit.split('_')[0]] * len(flist)
        
        if self.dataset_subset_portion < 1.0:
            import random
            random.shuffle(flist_all)
            num_total = len(flist_all)
            num_chosen = int(self.dataset_subset_portion*num_total)
            flist_all = flist_all[:num_chosen]
            print('Total examples: {}, now only randomly sample {} from them...'.format(num_total, num_chosen))

        self.scan_n, self.scan_pc = [], []
        self.body_verts = []
        self.vtransf = []

        for idx, fn in enumerate(tqdm(flist_all)):
            dd = np.load(fn)
            clo_type = dirname(fn).split('/')[-2] # e.g. rp_aaron_posed_002

            self.body_verts.append(torch.tensor(dd['body_verts']).float())
            self.scan_pc.append(torch.tensor(dd['scan_pc']).float())
            self.scan_n.append(torch.tensor(dd['scan_n']).float())
            
            vtransf = torch.tensor(dd['vtransf']).float()
            if vtransf.shape[-1] == 4:
                vtransf = vtransf[:, :3, :3]
            self.vtransf.append(vtransf)
    
    def __getitem__(self, index):
        body_verts = self.body_verts[index]
        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]

        vtransf = self.vtransf[index]

        # if self.scan_npoints != -1: 
        #     selected_idx = torch.randperm(len(scan_n))[:self.scan_npoints]
        #     scan_pc = scan_pc[selected_idx, :]
        #     scan_n = scan_n[selected_idx, :]

        return self.vertices, self.indices, self.barycentric_coords, \
            body_verts, scan_pc, scan_n, vtransf, index
    
    def __len__(self):
        return len(self.scan_n)


def test_smpl():
    smpl_face_path = '../assets/smplx_faces.npy'
    faces = np.load(smpl_face_path, allow_pickle=True)
    
    smpl_info_path = '../assets/smplx/SMPLX_NEUTRAL.npz'
    smpl_info = np.load(smpl_info_path, allow_pickle=True)
    
    smpl_info = dict(smpl_info)
    # ['hands_meanr', 'hands_meanl', 'lmk_bary_coords', 'vt', 
    # 'posedirs', 'part2num', 'hands_coeffsr', 'lmk_faces_idx', 
    # 'J_regressor', 'dynamic_lmk_faces_idx', 'hands_componentsr', 
    # 'shapedirs', 'dynamic_lmk_bary_coords', 'ft', 'hands_componentsl', 
    # 'joint2num', 'v_template', 'allow_pickle', 'f', 'hands_coeffsl', 
    # 'kintree_table', 'weights']
    # print(list(smpl_model.keys()))
    
    smpl_model_path = '../data/resynth/packed/rp_anna_posed_001/train/rp_anna_posed_001.96_jerseyshort_ATUsquat.00020.npz'
    # smpl_model = pickle.load(open(smpl_model_path, 'rb'), encoding='latin1')
    smpl_model = np.load(smpl_model_path, allow_pickle=True)
    smpl_model = dict(smpl_model)
    vertices = smpl_model['body_verts']
    
    target_pc = smpl_model['scan_pc']
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target_pc)
    )
    o3d.io.write_point_cloud('smplx_anna.ply', pcd)
    
    # vertices = smpl_model['v_template']
    # weights = smpl_model['weights']
    # posedirs = smpl_model['posedirs']
    
    # print(type(posedirs))  # (10475, 3, 486)
    # print(posedirs.shape)  # (10475, 55)
    
    # print(vertices.shape)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces)
    )
    
    # o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh('smplx_test.obj', mesh)
    
    # pcd = mesh.sample_points_uniformly(number_of_points=20000)
    # o3d.io.write_point_cloud('smplx_test_20000.ply', pcd)
    # points = np.asarray(pcd.points)


def test_resynth_data():
    import os
    
    data_dir = '../data/packed'
    human = 'rp_aaron_posed_002'
    split = 'train'
    test_file = 'rp_aaron_posed_002.96_jerseyshort_ATUsquat.00172.npz'
    
    file_path = os.path.join(data_dir, human, split, test_file)
    
    model = dict(np.load(file_path, allow_pickle=True))
    
    # print(list(model.keys()))
    body_verts = model['body_verts']
    scan_pc = model['scan_pc']
    scan_n = model['scan_n']
    vtransf = model['vtransf']

    # print(body_verts.shape)  # (10475, 3)
    # print(scan_pc.shape)    # (40000, 3)
    # print(scan_n.shape)     # (40000, 3)
    
    # print(vtransf.shape)    # (10475, 3, 3)
    
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(body_verts)
    )
    o3d.io.write_point_cloud('body_verts.ply', pcd)


def test_scipy_kdtree():
    cloud_A = np.random.rand(2, 3)
    cloud_B = np.random.rand(3, 3)
    
    # build a KDTree for cloud B
    tree_B = KDTree(cloud_B)
    
    # find the 3 closest points in cloud B for each point in cloud A
    distances, indices = tree_B.query(cloud_A, k=3)
    
    print("cloud_A: ", cloud_A)
    print("cloud_B: ", cloud_B)
    
    print("=========================")
    
    print("distances: ", distances)
    print("indices: ", indices)
    
    print("=========================")
    
    for pointA in cloud_A:
        for pointB in cloud_B:
            dist = np.linalg.norm(pointA - pointB)
            print(dist, end=" ")
        print()


def test_scan_pcl():
    
    data_path = '../data/resynth/scans/rp_felice_posed_004/'
    
    
    
    pass


if __name__ == '__main__':
    test_smpl()
    # test_resynth_data()
    # test_scipy_kdtree()
    
    # test SkiRT dataset
    # dataset = SKiRTCoarseDataset(
    #     dataset_type='resynth',
    #     data_root='../data/packed', 
    #     dataset_subset_portion=1.0,
    #     sample_spacing = 1,
    #     outfits={'rp_aaron_posed_002'},
    #     smpl_face_path='../assets/smplx_faces.npy',
    #     smpl_model_path='../assets/smplx/SMPLX_NEUTRAL.pkl',
    #     split='train', 
    #     num_samples=20000,
    #     knn=3,
    #     body_model='smplx'
    # )