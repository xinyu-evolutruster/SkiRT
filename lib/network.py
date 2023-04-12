import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_model import PositionalEncoding, normalize_uv, uv_to_grid
from lib.modules import UnetNoCond5DS, UnetNoCond6DS, UnetNoCond7DS, GeomConvLayers, GaussianSmoothingLayers, GeomConvBottleneckLayers, ShapeDecoder

from lib.utils_model import sample_and_group, sample_and_group_all, square_distance, index_points


class POP(nn.Module):
    def __init__(
                self, 
                input_nc=3, # num channels of the unet input
                c_geom=64, # channels of the geometric features
                c_pose=64, # channels of the pose features
                geom_layer_type='conv', # the type of architecture used for smoothing the geometric feature tensor
                nf=64, # num filters for the unet
                inp_posmap_size=128, # size of UV positional map (pose conditioning), i.e. the input to the pose unet
                hsize=256, # hidden layer size of the ShapeDecoder MLP
                up_mode='upconv', # upconv or upsample for the upsampling layers in the pose feature UNet
                use_dropout=False, # whether use dropout in the pose feature UNet
                pos_encoding=False, # whether use Positional Encoding for the query UV coordinates
                num_emb_freqs=8, # number of sinusoida frequences if positional encoding is used
                posemb_incl_input=False, # wheter include the original coordinate if using Positional Encoding
                uv_feat_dim=2, # input dimension of the uv coordinates
                pq_feat_dim = 2, # input dimension of the pq coordinates
                gaussian_kernel_size = 3, #optional, if use gaussian smoothing for the geometric features
                ):

        super().__init__()
        self.inp_posmap_size = inp_posmap_size
        self.pos_encoding = pos_encoding
        self.geom_layer_type = geom_layer_type 
        
        geom_proc_layers = {
            'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout), # use a unet
            'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'gaussian': GaussianSmoothingLayers(channels=c_geom, kernel_size=gaussian_kernel_size, sigma=1.0), # use a fixed gaussian smoother
        }
        unets = {32: UnetNoCond5DS, 64: UnetNoCond6DS, 128: UnetNoCond7DS, 256: UnetNoCond7DS}
        unet_loaded = unets[self.inp_posmap_size]

        if self.pos_encoding:
            self.embedder = PositionalEncoding(num_freqs=num_emb_freqs,
                                               input_dims=uv_feat_dim,
                                               include_input=posemb_incl_input)
            self.embedder.create_embedding_fn()
            self.uv_feat_dim = self.embedder.out_dim

        # U-net: for extracting pixel-aligned pose features from the input UV positional maps
        self.unet_posefeat = unet_loaded(input_nc, c_pose, nf, up_mode=up_mode, use_dropout=use_dropout)

        # optional layer for spatially smoothing the geometric feature tensor
        if geom_layer_type is not None:
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]
    
        # shared shape decoder across different outfit types
        self.decoder = ShapeDecoder(in_size=uv_feat_dim + pq_feat_dim + c_geom + c_pose,
                                    hsize=hsize, actv_fn='softplus')

            
    def forward(self, x, geom_featmap, uv_loc, pq_coords):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].
        :param pq_coords: the 'sub-UV-pixel' (p,q) coordinates, range [0,1), shape [B, H*W, 1, 2]. 
                        Note: It is the intra-patch coordinates in SCALE. Kept here for the backward compatibility with SCALE.
        :return:
            clothing offset vectors (residuals) and normals of the points
        '''
        # pose features
        pose_featmap = self.unet_posefeat(x)

        # geometric feature tensor
        if self.geom_layer_type is not None:
            geom_featmap = self.geom_proc_layers(geom_featmap)

        # pose and geom features are concatenated to form the feature for each point
        pix_feature = torch.cat([pose_featmap, geom_featmap], 1)

        feat_res = geom_featmap.shape[2] # spatial resolution of the input pose and geometric features
        uv_res = int(uv_loc.shape[1]**0.5) # spatial resolution of the query uv map

        # spatially bilinearly upsample the features to match the query resolution
        if feat_res != uv_res:
            query_grid = uv_to_grid(uv_loc, uv_res)
            pix_feature = F.grid_sample(pix_feature, query_grid, mode='bilinear', align_corners=False)#, align_corners=True)

        B, C, H, W = pix_feature.shape
        N_subsample = 1 # inherit the SCALE code custom, but now only sample one point per pixel

        uv_feat_dim = uv_loc.size()[-1]
        pq_coords = pq_coords.reshape(B, -1, 2).transpose(1, 2)  # [B, 2, Num all subpixels]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])

        # uv and pix feature is shared for all points in each patch
        pix_feature = pix_feature.view(B, C, -1).expand(N_subsample, -1,-1,-1).permute([1,2,3,0]) # [B, C, N_pix, N_sample_perpix]
        pix_feature = pix_feature.reshape(B, C, -1)

        if self.pos_encoding:
            uv_loc = normalize_uv(uv_loc).view(-1,uv_feat_dim)
            uv_loc = self.embedder.embed(uv_loc).view(B, -1,self.embedder.out_dim).transpose(1,2)
        else:
            uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(1, 2)  # [B, N_pix, N_subsample, 2] --> [B, 2, Num of all pq subpixels]

        residuals, normals = self.decoder(torch.cat([pix_feature, uv_loc, pq_coords], 1))  # [B, 3, N all subpixels]

        residuals = residuals.view(B, 3, H, W, N_subsample)
        normals = normals.view(B, 3, H, W, N_subsample)
        
        return residuals, normals


class SkiRTCoarseNetwork(nn.Module):
    def __init__(
        self,
        input_nc=3,    # num channels of the input point
        input_sc=256,  # num channels of the shape code
        num_layers=5,   # num of the MLP layers
        num_layers_loc=3, # num of the MLP layers for the location prediction
        num_layers_norm=3, # num of the MLP layers for the normal prediction
        hsize=256,     # hidden layer size of the MLP
        skip_layer=[4],  # skip layers
        actv_fn='softplus',  # activation function
        pos_encoding=False,   # whether use Positional Encoding for the query point
        num_emb_freqs=4,  # number of frequencies for the positional encoding
    ):
        super().__init__()
        
        self.input_nc = input_nc
        self.input_sc = input_sc
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.embedder = PositionalEncoding(
                input_dims=self.input_nc, num_freqs=num_emb_freqs)
            self.embedder.create_embedding_fn()
            self.input_nc = self.embedder.out_dim
        
        self.skip_layers = skip_layer
        
        self.actv_fn = None
        if actv_fn == 'relu':
            self.actv_fn = nn.ReLU()
        elif actv_fn == 'leaky_relu':
            self.actv_fn = nn.LeakyReLU()
        else:
            self.actv_fn = nn.Softplus()
        
        self.conv1 = nn.Conv1d(in_channels=self.input_nc+self.input_sc, out_channels=hsize, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hsize, out_channels=hsize, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=hsize, out_channels=hsize, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=hsize+self.input_nc+self.input_sc, out_channels=hsize, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=hsize, out_channels=hsize, kernel_size=1)
        
        self.bn1 = nn.BatchNorm1d(hsize)
        self.bn2 = nn.BatchNorm1d(hsize)
        self.bn3 = nn.BatchNorm1d(hsize)
        self.bn4 = nn.BatchNorm1d(hsize)
        self.bn5 = nn.BatchNorm1d(hsize)
        
        # point locations prediction branch
        self.point_loc_layers = []
        for layer in range(num_layers_loc-1):
            self.point_loc_layers.append(nn.Conv1d(
                in_channels=hsize, out_channels=hsize, kernel_size=1))
            self.point_loc_layers.append(nn.BatchNorm1d(hsize))
            self.point_loc_layers.append(self.actv_fn)
        self.point_loc_layers.append(nn.Conv1d(in_channels=hsize, out_channels=3, kernel_size=1))
        
        # point normals prediction branch
        self.point_norm_layers = []
        for layer in range(num_layers_norm-1):
            self.point_norm_layers.append(nn.Conv1d(
                in_channels=hsize, out_channels=hsize, kernel_size=1))
            self.point_norm_layers.append(nn.BatchNorm1d(hsize))
            self.point_norm_layers.append(self.actv_fn)
        self.point_norm_layers.append(nn.Conv1d(in_channels=hsize, out_channels=3, kernel_size=1))
        
        self.point_loc_layers = nn.Sequential(*self.point_loc_layers)
        self.point_norm_layers = nn.Sequential(*self.point_norm_layers)
        
    
    def forward(self, point_coord, shape_code):
        """
        Input:
            point_coord: the Cartesian coordinates of a query point 
            from the surface of the SMPL-X model. 
            - shape: [B, N, 3]
            shape_code: the global shape code of the SMPL-X model shared 
            by every query locations.
            - shape: [1, 256]
        Output:
            point_locations: the predicted 3D locations of the query points.
            point_normals: the predicted 3D normals of the query points.
        """
        
        B, N = point_coord.shape[:2]
        if self.pos_encoding:
            point_coord = self.embedder.embed(point_coord)
        
        shape_code = shape_code.unsqueeze(-1).repeat(B, 1, point_coord.shape[1]) # [B, 256, N]
        point_coord = point_coord.transpose(1, 2) # [B, 3, N]
        
        input_feat = torch.cat([point_coord, shape_code], 1) # [B, 259, N]
        
        out_feat = self.actv_fn(self.bn1(self.conv1(input_feat)))
        out_feat = self.actv_fn(self.bn2(self.conv2(out_feat)))
        out_feat = self.actv_fn(self.bn3(self.conv3(out_feat)))
        out_feat = torch.cat([out_feat, input_feat], 1)
        out_feat = self.actv_fn(self.bn4(self.conv4(out_feat)))
        out_feat = self.actv_fn(self.bn5(self.conv5(out_feat)))
    
        pred_residuals = self.point_loc_layers(out_feat).permute(0, 2, 1)  # [B, N, 3]
        pred_normals = self.point_norm_layers(out_feat).permute(0, 2, 1)   # [B, N, 3]
        
        return pred_residuals, pred_normals


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp=None, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.group_all = group_all
        
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        
        if points is not None:
            points = points.permute(0, 2, 1)
            
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, dim=2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


class PoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128, mlp=[128, 128, 256], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1)
        
    def forward(self, xyz, points):
        
        xyz = xyz.permute(0, 2, 1)  # [B, C, N]
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        feat = self.conv2(feat)
        
        return feat


class GeometryEncoder(nn.Module):
    def __init__(self, input_channel=64, hidden_size=128, output_channel=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(input_channel, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, output_channel, kernel_size=1),
        )
        
    def forward(self, x):
        feat = self.layers(x)
        return feat


class SkiRTShapeDecoder(nn.Module):
    '''
    The "Shape Decoder" in the POP paper Fig. 2. The same as the "shared MLP" in the SCALE paper.
    - with skip connection from the input features to the 4th layer's output features (like DeepSDF)
    - branches out at the second-to-last layer, one branch for position pred, one for normal pred
    '''
    def __init__(self, in_size, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(SkiRTShapeDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, 3, 1)

        self.conv6N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7N = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8N = torch.nn.Conv1d(self.hsize, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.bn6N = torch.nn.BatchNorm1d(self.hsize)
        self.bn7N = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # normals pred
        xN6 = self.actv_fn(self.bn6N(self.conv6N(x5)))
        xN7 = self.actv_fn(self.bn7N(self.conv7N(xN6)))
        xN8 = self.conv8N(xN7)

        return x8, xN8
    
    
class SkiRTFineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.pose_feature_encoder = PoseEncoder()
        self.geom_feature_encoder = GeometryEncoder()

        self.decoder = SkiRTShapeDecoder(in_size=128)
    
    def forward(self, xyz, points, geom_feat):
        
        pose_feature = self.pose_feature_encoder(xyz, points)
        geom_feature = self.geom_feature_encoder(geom_feat)
        
        residuals, normals = torch.cat([pose_feature, geom_feature], dim=1)
        
        return residuals, normals