import os
from os.path import join, dirname
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import open3d as o3d
import pickle
import trimesh
from scipy.spatial import KDTree

from lib.utils_model import sample_points

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class SKiRTCoarseDataset(Dataset):
    def __init__(
        self, 
        dataset_type='resynth',
        data_root=None, 
        dataset_subset_portion=1.0,
        sample_spacing = 1,
        outfits={},
        smpl_model_path=None,
        split='train', 
        num_samples=20000,
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
            mesh = trimesh.exchange.load.load(smpl_model_path)
            self.smpl_sampled_points, self.indices, self.bary_coords = sample_points(mesh, num_samples)
        else:
            raise NotImplementedError('Only support SMPLX model for now.')

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

        return self.smpl_sampled_points, self.indices, self.bary_coords, \
            body_verts, scan_pc, scan_n, vtransf, index
    
    def __len__(self):
        return len(self.scan_n)
    
    
class SKiRTFineDataset(Dataset):
    def __init__(
        self, 
        dataset_type='resynth',
        data_root=None, 
        dataset_subset_portion=1.0,
        sample_spacing = 1,
        outfits={},
        smpl_model_path=None,
        smpl_face_path=None,
        split='train', 
        num_samples=20000,
        body_model='smplx'):
        
        self.dataset_type = dataset_type
        self.data_root = data_root
        self.split = split
        self.spacing = sample_spacing
        self.num_samples = num_samples
        
        # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.data_dirs = {outfit: join(data_root, outfit, split) for outfit in outfits} 
        
        # randomly subsample a number of data from each clothing type (using all data from all outfits will be too much)
        self.dataset_subset_portion = dataset_subset_portion 
        
        # load smpl faces
        self.faces = np.load(smpl_face_path, allow_pickle=True)
        
        # import smplx model
        self.smpl_sampled_points = None
        self.faces, self.vertices = None, None
        if body_model == 'smplx':
            mesh = trimesh.exchange.load.load(smpl_model_path)
            self.smpl_sampled_points, self.indices, self.bary_coords = sample_points(mesh, num_samples)
        else:
            raise NotImplementedError('Only support SMPLX model for now.')
        
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
        """
        Every time we sample a new random point cloud from the SMPL body.
        """
        body_verts = self.body_verts[index]
        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]

        vtransf = self.vtransf[index]

        # mesh = trimesh.Trimesh(body_verts, self.faces, process=False)
        # smpl_sampled_points, indices, bary_coords = sample_points(mesh, self.num_samples)

        return self.smpl_sampled_points, self.indices, self.bary_coords, \
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
    # test_smpl()
    # test_resynth_data()
    # test_scipy_kdtree()
    
    # test SkiRT dataset
    dataset = SKiRTCoarseDataset(
        dataset_type='resynth',
        data_root='../data/packed', 
        dataset_subset_portion=1.0,
        sample_spacing = 1,
        outfits={'rp_aaron_posed_002'},
        smpl_face_path='../assets/smplx_faces.npy',
        smpl_model_path='../assets/smplx/SMPLX_NEUTRAL.pkl',
        split='train', 
        num_samples=20000,
        knn=3,
        body_model='smplx'
    )