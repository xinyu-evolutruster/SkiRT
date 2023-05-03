import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join, dirname

import trimesh
import open3d as o3d
from scipy.spatial import KDTree
from pytorch3d.ops import ball_query, knn_points

import torch
from torch.utils.data import Dataset

from lib.utils_model import sample_points

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# fix the random seed
torch.manual_seed(7)

class SKiRTCoarseDataset(Dataset):
    def __init__(
        self, 
        dataset_type='resynth',
        data_root=None, 
        dataset_subset_portion=1.0,
        sample_spacing = 1,
        outfits={},
        smpl_model_path=None,
        smpl_faces_path=None,
        split='train', 
        num_samples=20000,
        body_model='smplx'):
        
        self.dataset_type = dataset_type
        self.data_root = data_root
        self.split = split
        self.spacing = sample_spacing
        
        # will be sth like "./data/packed/cape/00032_shortlong/train"
        self.data_dirs = {outfit: join(data_root, outfit, split) for outfit in outfits} 
        
        print("self.data_dirs: ", self.data_dirs)
        
        # randomly subsample a number of data from each clothing type (using all data from all outfits will be too much)
        self.dataset_subset_portion = dataset_subset_portion 
        
        # import smplx model
        self.smpl_sampled_points = None
        self.smpl_faces = None
        if body_model == 'smplx':
            # load vertices
            smpl_info = np.load(smpl_model_path, allow_pickle=True)
            smpl_info = dict(smpl_info)
            self.smpl_sampled_points = torch.from_numpy(smpl_info['v_template'])
            # load faces
            self.smpl_faces = np.load(smpl_faces_path, allow_pickle=True)
        else:
            raise NotImplementedError('Only support SMPLX model for now.')

        # sample some random points
        self.random_sampled_points, self.rand_associated_indices = self.sample_random_points(
            smpl_sampled_points=self.smpl_sampled_points,
            smpl_faces=self.smpl_faces,
            threshold=0.15, num_samples=40000
        )

        self.num_smpl_points = self.smpl_sampled_points.shape[0]
        self.num_rand_points = self.random_sampled_points.shape[0]
        
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
    
    def sample_random_points(self, smpl_sampled_points, smpl_faces, threshold=0.15, num_samples=20000):
        # load mesh and convert to open3d geometry
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(smpl_sampled_points),
            o3d.utility.Vector3iVector(smpl_faces)
        )
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        
        xmin, ymin, zmin = torch.min(self.smpl_sampled_points, axis=0)[0]
        xmax, ymax, zmax = torch.max(self.smpl_sampled_points, axis=0)[0]
        zmin = zmin - threshold
        zmax = zmax + threshold
        
        num_xpoints, num_ypoints, num_zpoints = int((xmax-xmin)*30), int((ymax-ymin)*30), int((zmax-zmin)*30)
        num_points = num_xpoints * num_ypoints * num_zpoints
        
        x = np.linspace(xmin, xmax, num_xpoints, endpoint=False) + 1/(2*num_xpoints)
        y = np.linspace(ymin, ymax, num_ypoints, endpoint=False) + 1/(2*num_ypoints)
        z = np.linspace(zmin, zmax, num_zpoints, endpoint=False) + 1/(2*num_zpoints)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        cell_centers = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        query_points = o3d.core.Tensor(cell_centers, dtype=o3d.core.Dtype.Float32)
        
        # calculate the sdf of mesh
        distances = scene.compute_signed_distance(query_points)
        distances = np.asarray(distances)
        
        valid_points = cell_centers[distances<threshold]
        distances = distances[distances<threshold]
        valid_points = torch.tensor(valid_points[distances>0]).float()
        
        print("ymin = {}, ymax = {}".format(ymin, ymax))
        
        mask = valid_points[:, 1] <= -0.2
        valid_points = valid_points[mask]
        mask = valid_points[:, 1] >= -0.7
        valid_points = valid_points[mask]

        distances, indices, _ = knn_points(
            valid_points.unsqueeze(dim=0), 
            self.smpl_sampled_points.unsqueeze(dim=0),
            K=40, return_nn=False
        )
        rand_associated_indices = indices.squeeze(dim=0)

        # find averaged points
        view_shape = list(indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(1, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        smpl_points = self.smpl_sampled_points.unsqueeze(dim=0)
        averaged_points = smpl_points[batch_indices, indices, :]
        averaged_points = torch.mean(averaged_points, dim=2)

        # find the nearest points on the mesh
        # if the distance is below a certain threshold: the point is too close to the body,
        # thus we discard it as the body mesh itself will suffice to model the clothes.
        distances, _, _ = knn_points(
            averaged_points, 
            smpl_points,
            K=1, return_nn=False
        )
        # print(distances.max())
        thresh = 8e-4
        
        mask = (distances.squeeze(dim=0) > thresh)
        cell_centers = valid_points[mask.repeat(1,3)].view(-1, 3)
        
        num_valid_points = cell_centers.shape[0]
        rand_points = np.zeros((num_valid_points*10, 3))        
        side_len = 2.2
        for j in range(10):    
            for i in range(num_valid_points):
                x, y, z = cell_centers[i]
                rand_points[j*num_valid_points+i] = np.random.uniform(
                    [x-side_len/(2*num_xpoints), y-side_len/(2*num_ypoints), z-side_len/(2*num_zpoints)], 
                    [x+side_len/(2*num_xpoints), y+side_len/(2*num_ypoints), z+side_len/(2*num_zpoints)]
                )
        
        rand_points = torch.tensor(rand_points).float()
        
        distances, indices, _ = knn_points(
            rand_points.unsqueeze(dim=0), 
            self.smpl_sampled_points.unsqueeze(dim=0),
            K=40, return_nn=False
        )
        rand_associated_indices = indices.squeeze(dim=0)
        
        # find averaged points
        view_shape = list(indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(1, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        smpl_points = self.smpl_sampled_points.unsqueeze(dim=0)
        averaged_points = smpl_points[batch_indices, indices, :]
        averaged_points = torch.mean(averaged_points, dim=2)
        random_sampled_points = averaged_points.squeeze(dim=0)

        print("random_sampled_points.shape: ", random_sampled_points.shape)

        return random_sampled_points, rand_associated_indices
    
    def __getitem__(self, index):
        body_verts = self.body_verts[index]
        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]
        vtransf = self.vtransf[index]
        
        query_points = torch.cat((self.smpl_sampled_points, self.random_sampled_points), dim=0)
        
        return query_points, self.rand_associated_indices, body_verts, scan_pc, scan_n, vtransf, index
    
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
        num_samples=40000,
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
        
        # import smplx model
        self.smpl_sampled_points = None
        self.smpl_vertices, self.smpl_faces = None, None
        if body_model == 'smplx':
            # load vertices
            smpl_info = np.load(smpl_model_path, allow_pickle=True)
            smpl_info = dict(smpl_info)
            self.smpl_vertices = torch.from_numpy(smpl_info['v_template'])
            # load faces
            self.smpl_faces = np.load(smpl_face_path, allow_pickle=True)
            # create a trimesh mesh
            mesh = trimesh.Trimesh(vertices=self.smpl_vertices, faces=self.smpl_faces)
            self.smpl_sampled_points, self.indices, self.bary_coords = sample_points(mesh, n_points=num_samples)
            # print("type of self.smpl_sampled_points: ", type(self.smpl_sampled_points))
        else:
            raise NotImplementedError('Only support SMPLX model for now.')
        
        # sample some random points
        self.random_sampled_points, self.rand_associated_indices = self.sample_random_points(
            smpl_sampled_points=self.smpl_vertices,
            smpl_faces=self.smpl_faces,
            threshold=0.15, num_samples=40000
        )

        self.num_smpl_points = self.smpl_sampled_points.shape[0]
        self.num_rand_points = self.random_sampled_points.shape[0]
        
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
    
    def sample_random_points(self, smpl_sampled_points, smpl_faces, threshold=0.15, num_samples=20000):
        # load mesh and convert to open3d geometry
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(smpl_sampled_points),
            o3d.utility.Vector3iVector(smpl_faces)
        )
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        
        xmin, ymin, zmin = torch.min(self.smpl_vertices, axis=0)[0]
        xmax, ymax, zmax = torch.max(self.smpl_vertices, axis=0)[0]
        zmin = zmin - threshold
        zmax = zmax + threshold
        
        num_xpoints, num_ypoints, num_zpoints = int((xmax-xmin)*30), int((ymax-ymin)*30), int((zmax-zmin)*30)
        num_points = num_xpoints * num_ypoints * num_zpoints
        
        x = np.linspace(xmin, xmax, num_xpoints, endpoint=False) + 1/(2*num_xpoints)
        y = np.linspace(ymin, ymax, num_ypoints, endpoint=False) + 1/(2*num_ypoints)
        z = np.linspace(zmin, zmax, num_zpoints, endpoint=False) + 1/(2*num_zpoints)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        cell_centers = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        query_points = o3d.core.Tensor(cell_centers, dtype=o3d.core.Dtype.Float32)
        
        # calculate the sdf of mesh
        distances = scene.compute_signed_distance(query_points)
        distances = np.asarray(distances)
        
        valid_points = cell_centers[distances<threshold]
        distances = distances[distances<threshold]
        valid_points = torch.tensor(valid_points[distances>0]).float()
        
        print("ymin = {}, ymax = {}".format(ymin, ymax))
        
        mask = valid_points[:, 1] <= -0.2
        valid_points = valid_points[mask]
        mask = valid_points[:, 1] >= -0.7
        valid_points = valid_points[mask]

        distances, indices, _ = knn_points(
            valid_points.unsqueeze(dim=0), 
            self.smpl_vertices.unsqueeze(dim=0),
            K=40, return_nn=False
        )
        rand_associated_indices = indices.squeeze(dim=0)

        # find averaged points
        view_shape = list(indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(1, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        smpl_points = self.smpl_vertices.unsqueeze(dim=0)
        averaged_points = smpl_points[batch_indices, indices, :]
        averaged_points = torch.mean(averaged_points, dim=2)

        # find the nearest points on the mesh
        # if the distance is below a certain threshold: the point is too close to the body,
        # thus we discard it as the body mesh itself will suffice to model the clothes.
        distances, _, _ = knn_points(
            averaged_points, 
            smpl_points,
            K=1, return_nn=False
        )
        # print(distances.max())
        thresh = 8e-4
        
        mask = (distances.squeeze(dim=0) > thresh)
        cell_centers = valid_points[mask.repeat(1,3)].view(-1, 3)
        
        num_valid_points = cell_centers.shape[0]
        rand_points = np.zeros((num_valid_points*50, 3))        
        side_len = 2.5
        for j in range(10):    
            for i in range(num_valid_points):
                x, y, z = cell_centers[i]
                rand_points[j*num_valid_points+i] = np.random.uniform(
                    [x-side_len/(2*num_xpoints), y-side_len/(2*num_ypoints), z-side_len/(2*num_zpoints)], 
                    [x+side_len/(2*num_xpoints), y+side_len/(2*num_ypoints), z+side_len/(2*num_zpoints)]
                )
        
        rand_points = torch.tensor(rand_points).float()
        
        distances, indices, _ = knn_points(
            rand_points.unsqueeze(dim=0), 
            self.smpl_vertices.unsqueeze(dim=0),
            K=40, return_nn=False
        )
        rand_associated_indices = indices.squeeze(dim=0)
        
        # find averaged points
        view_shape = list(indices.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(indices.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(1, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        smpl_points = self.smpl_vertices.unsqueeze(dim=0)
        averaged_points = smpl_points[batch_indices, indices, :]
        averaged_points = torch.mean(averaged_points, dim=2)
        random_sampled_points = averaged_points.squeeze(dim=0)

        print("random_sampled_points.shape: ", random_sampled_points.shape)

        return random_sampled_points, rand_associated_indices
    
    def __getitem__(self, index):
        """
        Every time we sample a new random point cloud from the SMPL body.
        """
        body_verts = self.body_verts[index]
        scan_n = self.scan_n[index]
        scan_pc = self.scan_pc[index]
        vtransf = self.vtransf[index]

        query_points = torch.cat((self.smpl_sampled_points, self.random_sampled_points), dim=0)
        
        return query_points, self.indices, self.rand_associated_indices, \
            self.bary_coords, body_verts, scan_pc, scan_n, vtransf, index
    
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
    print(list(smpl_info.keys()))
    J_regressor = smpl_info['J_regressor']
    print("J_regressor shape: ", J_regressor.shape)
    weights = smpl_info['weights']
    print("weights shape: ", weights.shape)
    
    smpl_model_path = '../data/resynth/packed/rp_anna_posed_001/train/rp_anna_posed_001.96_jerseyshort_ATUsquat.00020.npz'
    # smpl_model = pickle.load(open(smpl_model_path, 'rb'), encoding='latin1')
    smpl_model = np.load(smpl_model_path, allow_pickle=True)
    smpl_model = dict(smpl_model)
    vertices = smpl_model['body_verts']
    
    # target_pc = smpl_model['scan_pc']
    # pcd = o3d.geometry.PointCloud(
    #     o3d.utility.Vector3dVector(target_pc)
    # )
    # o3d.io.write_point_cloud('smplx_anna.ply', pcd)
    
    # vertices = smpl_model['v_template']
    # weights = smpl_model['weights']
    # posedirs = smpl_model['posedirs']
    
    # print(type(posedirs))  # (10475, 3, 486)
    # print(posedirs.shape)  # (10475, 55)
    
    # print(vertices.shape)
    # mesh = o3d.geometry.TriangleMesh(
    #     o3d.utility.Vector3dVector(vertices),
    #     o3d.utility.Vector3iVector(faces)
    # )
    
    # o3d.visualization.draw_geometries([mesh])
    # o3d.io.write_triangle_mesh('smplx_test.obj', mesh)
    
    # pcd = mesh.sample_points_uniformly(number_of_points=20000)
    # o3d.io.write_point_cloud('smplx_test_20000.ply', pcd)
    # points = np.asarray(pcd.points)


def test_resynth_data():
    import os
    
    data_dir = '../data/resynth/packed'
    human = 'rp_aaron_posed_002'
    split = 'train'
    test_file = 'rp_aaron_posed_002.96_jerseyshort_ATUsquat.00172.npz'
    
    file_path = os.path.join(data_dir, human, split, test_file)
    
    model = dict(np.load(file_path, allow_pickle=True))
    
    print(list(model.keys()))
    body_verts = model['body_verts']
    scan_pc = model['scan_pc']
    scan_n = model['scan_n']
    vtransf = model['vtransf']

    # print(body_verts.shape)  # (10475, 3)
    # print(scan_pc.shape)    # (40000, 3)
    # print(scan_n.shape)     # (40000, 3)
    
    # print(vtransf.shape)    # (10475, 3, 3)
    
    # pcd = o3d.geometry.PointCloud(
    #     o3d.utility.Vector3dVector(body_verts)
    # )
    # o3d.io.write_point_cloud('body_verts.ply', pcd)


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


def test_random_sampling():
    from utils_io import customized_export_ply
    path = '../assets/smplx/SMPLX_NEUTRAL.npz'
    smpl_face_path = '../assets/smplx_faces.npy'
    
    # load mesh and convert to open3d geometry
    smpl_info = np.load(path, allow_pickle=True)
    smpl_info = dict(smpl_info)
    
    smpl_sampled_points = smpl_info['v_template']
    faces = np.load(smpl_face_path, allow_pickle=True)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(smpl_sampled_points),
        o3d.utility.Vector3iVector(faces)
    )
    
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    
    num_samples = 20000
    
    xmin, ymin, zmin = np.min(smpl_sampled_points, axis=0)
    xmax, ymax, zmax = np.max(smpl_sampled_points, axis=0)
        
    points = np.random.rand(num_samples, 3)
    x = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
    y = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
    z = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
    x = xmin + (xmax - xmin) * x
    y = ymin + (ymax - ymin) * y
    z = zmin + (zmax - zmin) * z
    rand_points = np.stack([x, y, z], axis=1)
    
    query_points = o3d.core.Tensor(rand_points, dtype=o3d.core.Dtype.Float32)
    
    # calculate the sdf of mesh
    distances = scene.compute_signed_distance(query_points)
    distances = np.asarray(distances)
    
    threshold = 0.25
    valid_points = rand_points[distances<threshold]
    distances = distances[distances<threshold]
    valid_points = valid_points[distances>0]
    
    mask = valid_points[:, 1] <= -0.5
    valid_points = valid_points[mask]
    
    all_points = np.concatenate([smpl_sampled_points, valid_points], axis=0)
    
    red = np.array([255, 0, 0]).reshape(1, 3).repeat(10475, axis=0)
    green = np.array([0, 255, 0]).reshape(1, 3).repeat(valid_points.shape[0], axis=0)
    vc = np.concatenate([red, green], axis=0)
    
    valid_points = torch.from_numpy(valid_points).float()
    smpl_sampled_points = torch.from_numpy(np.asarray(smpl_sampled_points)).float()
    
    customized_export_ply(
        'query_points.ply',
        v=all_points,
        v_c=vc
    )
    
    distances, indices, _ = knn_points(
        valid_points.unsqueeze(dim=0), 
        smpl_sampled_points.unsqueeze(dim=0),
        K=40, return_nn=False
    )
    # weights = 1.0 / (distances + 1e-6)
    # # normalize weights
    # weights = weights / torch.sum(weights, dim=2, keepdim=True)
    
    B, _ = indices.shape[:2]
    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indices.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    averaged_points = smpl_sampled_points.unsqueeze(dim=0).repeat(B, 1, 1)[batch_indices, indices, :]
    averaged_points = torch.mean(averaged_points, dim=2)
    
    print("averaged_points: ", averaged_points.shape)
    
    customized_export_ply(
        'smplx_sampled_points.ply',
        v=averaged_points[0],
    )   

    # try body verts
    # test_file = 'rp_aaron_posed_002.96_jerseyshort_tilt_twist_left.00064.npz'
    test_file = 'rp_aaron_posed_002.96_jerseyshort_ATUsquat.00080.npz'
    data_dir = '../data/resynth/packed'
    human = 'rp_aaron_posed_002'
    split = 'train'
    file_path = os.path.join(data_dir, human, split, test_file)
    
    model = dict(np.load(file_path, allow_pickle=True))
    body_verts = torch.from_numpy(model['body_verts'])

    # mesh = o3d.geometry.TriangleMesh(
    #     o3d.utility.Vector3dVector(body_verts),
    #     o3d.utility.Vector3iVector(faces)
    # )
    # o3d.io.write_triangle_mesh('smplx_test.obj', mesh)
    # mesh = o3d.io.read_triangle_mesh('smplx_test.obj')
    # body_verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
    
    posed_averaged_points = body_verts.unsqueeze(dim=0).repeat(B, 1, 1)[batch_indices, indices, :]
    posed_averaged_points = torch.mean(posed_averaged_points, dim=2)
    
    customized_export_ply(
        'posed_averaged_points.ply',
        v=posed_averaged_points[0],
    )  


if __name__ == '__main__':
    # test_smpl()
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
    
    test_random_sampling()