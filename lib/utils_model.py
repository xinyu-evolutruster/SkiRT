import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
import open3d as o3d

import pytorch3d.ops.sample_farthest_points as sample_farthest_points
import pytorch3d.ops.ball_query as ball_query

def gen_transf_mtx_full_uv(verts, faces):
    '''
    given a positional uv map, for each of its pixel, get the matrix that transforms the prediction from local to global coordinates
    The local coordinates are defined by the posed body mesh (consists of vertcs and faces)

    :param verts: [batch, Nverts, 3]
    :param faces: [uv_size, uv_size, 3], uv_size =e.g. 32
    
    :return: [batch, uv_size, uv_size, 3,3], per example a map of 3x3 rot matrices for local->global transform

    NOTE: local coords are NOT cartesian! uu an vv axis are edges of the triangle,
          not perpendicular (more like barycentric coords)
    '''
    tris = verts[:, faces] # [batch, uv_size, uv_size, 3, 3]
    v1, v2, v3 = tris[:, :, :, 0, :], tris[:, :, :, 1, :], tris[:, :, :, 2, :]
    uu = v2 - v1 # u axis of local coords is the first edge, [batch, uv_size, uv_size, 3]
    vv = v3 - v1 # v axis, second edge
    ww_raw = torch.cross(uu, vv, dim=-1)
    ww = F.normalize(ww_raw, p=2, dim=-1) # unit triangle normal as w axis
    ww_norm = (torch.norm(uu, dim=-1).mean(-1).mean(-1) + torch.norm(vv, dim=-1).mean(-1).mean(-1)) / 2.
    ww = ww*ww_norm.view(len(ww_norm),1,1,1)
    
    # shape of transf_mtx will be [batch, uv_size, uv_size, 3, 3], where the last two dim is like:
    #  |   |   |
    #[ uu  vv  ww]
    #  |   |   |
    # for local to global, say coord in the local coord system is (r,s,t)
    # then the coord in world system should be r*uu + s*vv+ t*ww
    # so the uu, vv, ww should be colum vectors of the local->global transf mtx
    # so when stack, always stack along dim -1 (i.e. column)
    transf_mtx = torch.stack([uu, vv, ww], dim=-1)

    return transf_mtx


def gen_transf_mtx_from_vtransf(vtransf, bary_coords, faces, scaling=1.0):
    '''
    interpolate the local -> global coord transormation given such transformations defined on 
    the body verts (pre-computed) and barycentric coordinates of the query points from the uv map.

    Note: The output of this function, i.e. the transformation matrix of each point, is not a pure rotation matrix (SO3).
    
    args:
        vtransf: [batch, #verts, 3, 3] # per-vertex rotation matrix
        bary_coords: [uv_size, uv_size, 3] # barycentric coordinates of each query point (pixel) on the query uv map 
        faces: [uv_size, uv_size, 3] # the vert id of the 3 vertices of the triangle where each uv pixel locates

    returns: 
        [batch, uv_size, uv_size, 3, 3], transformation matrix for points on the uv surface
    '''
    #  
    vtransf_by_tris = vtransf[:, faces] # shape will be [batch, uvsize, uvsize, 3, 3, 3], where the the last 2 dims being the transf (pure rotation) matrices, the other "3" are 3 points of each triangle
    transf_mtx_uv_pts = torch.einsum('bpqijk,pqi->bpqjk', vtransf_by_tris, bary_coords) # [batch, uvsize, uvsize, 3, 3], last 2 dims are the rotation matix
    transf_mtx_uv_pts *= scaling
    return transf_mtx_uv_pts


def get_transf_mtx_from_vtransf(vtransf, body_verts, bary_coords, indices):
    """
    Input:
        vtransf: [batch, 10475, 3, 3] # per-vertex rotation matrix
        body_verts: [batch, 10475, 3] # body verts
        indices: [batch, 20000, 3] # indices of the 3 vertices of the triangle where each point locates
    Output:
        transf_mtx: [batch, npoints, 3, 3], transformation matrix for points on the uv surface
    """
    
    B, V = vtransf.shape[:2]
    _, N = indices.shape[:2]
    
    idx = indices.view(B, -1)  # [batch, 3*npoints]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [batch, 1]
    repeat_shape = list(idx.shape) # [batch, 3]
    repeat_shape[0] = 1  # [1, 3]
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    
    # get transformation matrices
    vtransf = vtransf.view(B, V, -1)
    new_vtransf = vtransf[batch_indices, idx, :] # [batch, npoints*3, 3, 3]  
    new_vtransf = new_vtransf.view(B, N, 3, 3, 3) # [batch, npoints, 3, 3]
    
    transf_mtxs = bary_coords[:, :, 0].view(B, N, 1, 1) * new_vtransf[:, :, 0, :, :] + \
                 bary_coords[:, :, 1].view(B, N, 1, 1) * new_vtransf[:, :, 1, :, :] + \
                 bary_coords[:, :, 2].view(B, N, 1, 1) * new_vtransf[:, :, 2, :, :]
    
    # get body verts
    body_verts = body_verts[batch_indices, idx, :].view(B, N, 3, 3) # [batch, npoints, 3, 3]
    body_verts = bary_coords[:, :, 0].view(B, N, 1) * body_verts[:, :, 0, :] + \
                 bary_coords[:, :, 1].view(B, N, 1) * body_verts[:, :, 1, :] + \
                 bary_coords[:, :, 2].view(B, N, 1) * body_verts[:, :, 2, :]

    return transf_mtxs.double(), body_verts.double()


class SampleSquarePoints():
    def __init__(self, npoints=16, min_val=0, max_val=1, device='cuda', include_end=True):
        super(SampleSquarePoints, self).__init__()
        self.npoints = npoints
        self.device = device
        self.min_val = min_val # -1 or 0
        self.max_val = max_val # -1 or 0
        self.include_end = include_end

    def sample_regular_points(self, N=None):
        steps = int(self.npoints ** 0.5) if N is None else int(N ** 0.5)
        if self.include_end:
            linspace = torch.linspace(self.min_val, self.max_val, steps=steps) # [0,1]
        else:
            linspace = torch.linspace(self.min_val, self.max_val, steps=steps+1)[: steps] # [0,1)
        grid = torch.stack(torch.meshgrid([linspace, linspace]), -1).to(self.device) #[steps, steps, 2]
        grid = grid.view(-1,2).unsqueeze(0) #[B, N, 2]
        grid.requires_grad = True

        return grid

    def sample_random_points(self, N=None):
        npt = self.npoints if N is None else N
        shape = torch.Size((1, npt, 2))
        rand_grid = torch.Tensor(shape).float().to(self.device)
        rand_grid.data.uniform_(self.min_val, self.max_val)
        rand_grid.requires_grad = True #[B, N, 2]
        return rand_grid


class Embedder():
    '''
    Simple positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    '''
    Helper function for positional encoding, adapted from NeRF: https://github.com/bmild/nerf
    '''
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class PositionalEncoding():
    def __init__(self, input_dims=2, num_freqs=10, include_input=False):
        super(PositionalEncoding,self).__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dims = input_dims

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        freq_bands = 2. ** torch.linspace(0, self.num_freqs-1, self.num_freqs)
        periodic_fns = [torch.sin, torch.cos]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(math.pi * x * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,coords):
        '''
        use periodic positional encoding to transform cartesian positions to higher dimension
        :param coords: [N, 3]
        :return: [N, 3*2*num_freqs], where 2 comes from that for each frequency there's a sin() and cos()
        '''
        return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)


def normalize_uv(uv):
    '''
    normalize uv coords from range [0,1] to range [-1,1]
    '''
    return uv * 2. - 1.


def uv_to_grid(uv_idx_map, resolution):
    '''
    uv_idx_map: shape=[batch, N_uvcoords, 2], ranging between 0-1
    this function basically reshapes the uv_idx_map and shift its value range to (-1, 1) (required by F.gridsample)
    the sqaure of resolution = N_uvcoords
    '''
    bs = uv_idx_map.shape[0]
    grid = uv_idx_map.reshape(bs, resolution, resolution, 2) * 2 - 1.
    grid = grid.transpose(1,2)
    return grid


def sample_points(mesh, n_points):
    samples, face_index = trimesh.sample.sample_surface_even(mesh, n_points)

    # sample points in each triangle
    # ref: https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
    bary_coords = []
    face_indices = []
    for i, face in enumerate(face_index):
        v0, v1, v2 = mesh.faces[face]
        p0, p1, p2 = mesh.vertices[v0], mesh.vertices[v1], mesh.vertices[v2]
        p = samples[i]
        face_indices.append([v0, v1, v2])
        
        A = np.array([[p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [1, 1, 1]])
        b = np.array([p[0], p[1], 1])
        u, v, w = np.linalg.solve(A, b)
        bary_coords.append([u, v, w])
    
    face_indices = np.array(face_indices)
    bary_coords = np.array(bary_coords)
    
    samples = torch.from_numpy(samples).float()
    
    return samples, face_indices, bary_coords


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m

    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device 
    B = points.shape[0]
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    return new_points


def sample_and_group(npoints, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape[:3]
    S = npoints
    
    _, fps_idx = sample_farthest_points(xyz, K=S)  # [B, S]
    
    new_xyz = index_points(xyz, fps_idx)  # [B, S, C]
    
    new_xyz, xyz = new_xyz.float(), xyz.float()
    
    _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius)  # [B, S, nsample]
    
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)  # [B, S, nsample, C]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, S, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    
    B, N, C = xyz.shape[:3]
    new_xyz = torch.zeros(B, 1, C).to(device)  # [B, 1, C]
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # [B, 1, N, C+D]
    else:
        new_points = grouped_xyz
    
    return new_xyz, new_points


if __name__ == '__main__':
    mesh_path = '../outputs/smplx_test.obj'
    mesh = trimesh.exchange.load.load(mesh_path)
    
    sample_points(mesh, 20000)