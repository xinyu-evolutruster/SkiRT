import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_model import PositionalEncoding, normalize_uv, uv_to_grid
from lib.utils_model import sample_and_group, sample_and_group_all, square_distance, index_points

from lib.modules import UnetNoCond5DS, UnetNoCond6DS, UnetNoCond7DS, GeomConvLayers, GaussianSmoothingLayers, GeomConvBottleneckLayers, ShapeDecoder

from faster_pointnet2.pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule

# fix the random seed
torch.manual_seed(12345)

class STN3d(nn.Module):
    def __init__(self, in_channel):
        super(STN3d, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv1d(1024, 512, 1)
        self.fc2 = nn.Conv1d(512, 256, 1)
        self.fc3 = nn.Conv1d(256, 9, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        
    def forward(self, x):
        batchsize = x.size()[0]
    
        x = self.relu(self.bn1(self.conv1(x))) # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x))) # [B, 128, N]
        x = self.relu(self.bn3(self.conv3(x))) # [B, 1024, N]
        x = self.relu(self.bn4(self.fc1(x)))   # [B, 512, N]
        x = self.relu(self.bn5(self.fc2(x)))   # [B, 256, N]
        x = self.fc3(x)                        # [B, 9, N]
    
        x = x.view(batchsize, 3, 3, -1) # [B, 3, 3, N]
        x = x.permute(0, 3, 1, 2)        # [B, N, 3, 3]
        
        return x


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
        
        # [B, S, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
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
        xyz1 = xyz1.permute(0, 2, 1).float()  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1).float()  # [B, S, C]
        
        points2 = points2.permute(0, 2, 1)  # [B, S, D]
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
            points1 = points1.permute(0, 2, 1) # [B, N, D]            
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
             
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.resolution = int(resolution)
        self.normalize = normalize
        self.eps = eps
        
    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(2, dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1.0) / 2.0
        
        norm_coords = torch.clamp(norm_coords * self.resolution, 0, self.resolution - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.resolution, self.resolution), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.resolution, ', normalize eps = {}'.format(self.eps) if self.normalize else '')


class SE3d(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        return inputs * self.fc(inputs.mean((2, 3, 4), keepdim=True))


class SharedMLP(nn.Module):
    def __init__(self, in_channel, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        
        for out_channel in out_channels:
            layers.extend([
                conv(in_channel, out_channel, 1),
                bn(out_channel),
                nn.ReLU(inplace=True)
            ])
            in_channel = out_channel
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)

        return fused_features, coords


class PoseEncoder(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=16, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=16, in_channel=64+3, mlp=[64, 64, 128], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+3, mlp=[128, 128, 256], group_all=True)
        # self.fp3 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 256])
        # self.fp2 = PointNetFeaturePropagation(in_channel=256+64, mlp=[256, 256])
        # self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, 128, 128])
        
        self.sa1 = PointnetSAModule(
            mlp=[0, 32, 32, 64], 
            npoint=512, radius=0.1, nsample=16
        )
        self.sa2 = PointnetSAModule(
            mlp=[64, 64, 128],
            npoint=128, radius=0.2, nsample=16   
        )
        self.sa3 = PointnetSAModule(
            mlp=[128, 128, 256],
            npoint=None, radius=None, nsample=None
        )
        self.fp3 = PointnetFPModule(mlp=[384, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[320, 256, 256])
        self.fp1 = PointnetFPModule(mlp=[256, 128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=1)
        
    def forward(self, xyz, points):
        
        # xyz = xyz.permute(0, 2, 1)  # [B, C, N]
        # if points is not None:
        #     points = points.permute(0, 2, 1)  # [B, D, N]
        
        l0_points = points  # coarse input
        # l0_xyz = xyz[:, :3, :]
        l0_xyz = xyz
        
        l0_xyz = l0_xyz.contiguous().float()
        if l0_points is not None:
            l0_points = l0_points.contiguous().float()
        
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
        
        x = x.permute(0, 2, 1)  # [B, C, N]
        feat = self.layers(x)
        return feat


class SkiRTShapeDecoder(nn.Module):
    '''
    The "Shape Decoder" in the POP paper Fig. 2. The same as the "shared MLP" in the SCALE paper.
    - with skip connection from the input features to the 4th layer's output features (like DeepSDF)
    - branches out at the second-to-last layer, one branch for position pred, one for normal pred
    '''
    def __init__(self, in_size=64+64, hsize=128, actv_fn='softplus'):
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

        self.pose_feature_encoder = PoseEncoder(in_channel=3)
        self.geom_feature_encoder = GeometryEncoder()

        self.decoder = SkiRTShapeDecoder(in_size=128)
    
    def forward(self, xyz, points, geom_feat):
        
        B = xyz.shape[0]
        N, C = geom_feat.shape
        
        pose_feature = self.pose_feature_encoder(xyz, points)
        
        geom_feat = geom_feat.view(1, N, C).repeat(B, 1, 1)
        geom_feature = self.geom_feature_encoder(geom_feat)

        feature = torch.cat([pose_feature, geom_feature], dim=1)
        
        residuals, normals = self.decoder(feature)
        
        residuals = residuals.permute(0, 2, 1)
        normals = normals.permute(0, 2, 1)
        
        return residuals, normals
   

class FeatureExpansion(nn.Module):
    def __init__(self, in_size, hidden_size=64, out_size=64, r=9):
        super(FeatureExpansion, self).__init__()

        self.num_replicate = r
        self.expansions = nn.ModuleList()

        for i in range(self.num_replicate):
            mlp_convs = nn.ModuleList()
            mlp_convs.append(nn.Conv1d(in_size, hidden_size, kernel_size=1))
            mlp_convs.append(nn.BatchNorm1d(hidden_size))
            mlp_convs.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=1))
            mlp_convs.append(nn.BatchNorm1d(hidden_size))
            mlp_convs.append(nn.Conv1d(hidden_size, out_size, kernel_size=1))
            
            self.expansions.append(mlp_convs)

    def forward(self, x):
        output = None
        for i in range(self.num_replicate):
            mlp_convs = self.expansions[i]
            conv0 = mlp_convs[0]
            conv1 = mlp_convs[2]
            conv2 = mlp_convs[4]
            bn0 = mlp_convs[1]
            bn1 = mlp_convs[3]
            fea = F.relu(conv1(bn0(conv0(x))))
            fea = F.relu(conv2(bn1(fea)))

            if i == 0:
                output = fea.unsqueeze(1)
            else:
                fea = fea.unsqueeze(1)
                output = torch.cat([output, fea], dim=1)
             
        return output
 

class SkiRTUpsampleNetwork(nn.Module):
    def __init__(self, pose_feature_channel=64, geom_feature_channel=64, upsample_rate=4):
        super().__init__()

        self.pose_feature_channel=pose_feature_channel
        self.geom_feature_channel=geom_feature_channel
        self.upsample_rate = upsample_rate

        self.pose_feature_encoder = PoseEncoder(in_channel=3)
        self.geom_feature_encoder = GeometryEncoder()
        
        self.feature_expansion = FeatureExpansion(
            in_size=64, hidden_size=128, out_size=64, r=self.upsample_rate
        )

        self.decoder = SkiRTShapeDecoder(in_size=128)

    def forward(self, xyz, points, geom_feat, random_points):
        
        B, N = xyz.shape[0], xyz.shape[1]
        N2, C = geom_feat.shape

        xyz = torch.cat([xyz, random_points], dim=1)
        pose_feature = self.pose_feature_encoder(xyz, points)
        
        random_pose_feature = pose_feature[:, :, N:] 
        random_pose_feature = self.feature_expansion(random_pose_feature).permute([0, 1, 3, 2])
        random_pose_feature = random_pose_feature.reshape(B, self.pose_feature_channel, -1)
        
        # pose_feature = pose_feature.reshape(B, -1, self.pose_feature_channel)
        pose_feature = torch.cat([pose_feature[:, :, :N], random_pose_feature], dim=2).permute([0, 2, 1])
        
        geom_feat = geom_feat.view(1, N2, C).repeat(B, 1, 1)
        geom_feature = self.geom_feature_encoder(geom_feat).permute(0, 2, 1)
        
        random_geom_feature = geom_feature[:, N:, :].unsqueeze(1).repeat(1, self.upsample_rate, 1, 1)
        random_geom_feature = random_geom_feature.reshape(B, -1, self.geom_feature_channel)
        
        geom_feature = torch.cat([geom_feature[:, :N, :], random_geom_feature], dim=1)
        
        # geom_feature = geom_feature.unsqueeze(1).repeat(1, self.upsample_rate, 1, 1)
        # geom_feature = geom_feature.reshape(B, -1, self.geom_feature_channel)

        feature = torch.cat([pose_feature, geom_feature], dim=2).permute(0, 2, 1)
        
        residuals, normals = self.decoder(feature)
        
        residuals = residuals.permute(0, 2, 1)
        normals = normals.permute(0, 2, 1)
        
        return residuals, normals