"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
import imageio

from vis.utils import get_device, get_mesh_renderer, get_points_renderer
from vis.render_utils import render_turntable_pcl, render_turntable_mesh


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_point_cloud(
    verts, rgb,
    image_size=256,
    background_color=(1, 1, 1),
    point_radius=0.01,
    camera_dist=2,
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
        
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color, radius=point_radius
    )
    # verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    # rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=camera_dist, elev=0, azim=0, degrees=True)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    R1 = 1.0
    R2 = 0.3

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R1 + R2 * torch.cos(Theta)) * torch.cos(Phi)
    y = (R1 + R2 * torch.cos(Theta)) * torch.sin(Phi)
    z = R2 * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renders = render_turntable_pcl(
        sphere_point_cloud, num_frames=30, image_size=image_size, dist=3)
    imageio.mimsave("images/torus.gif", renders, fps=(30 // 3))

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()

    R1 = 1.0
    R2 = 0.3

    min_value = -1.6
    max_value = 1.6
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (R1 - torch.sqrt(X**2 + Y**2))**2 + Z**2 - R2 ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    renders = render_turntable_mesh(mesh, num_frames=30, image_size=image_size)
    imageio.mimsave("images/torus_implicit.gif", renders, fps=(30 // 3))

    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_heart(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()

    min_value = -1.6
    max_value = 1.6
    X, Y, Z = torch.meshgrid(
        [torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (X**2 + 9 / 4 * Z**2 + Y**2 - 1)**3 - \
        X**2 * Y**3 - 9/200 * Z**2 * Y**3
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    # textures = torch.tensor([1, 0, 0])
    # textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    color = (1, 0, 0)
    textures = torch.ones_like(vertices.unsqueeze(0))  # (1, N_v, 3)
    textures = textures * torch.tensor(color)
    textures = pytorch3d.renderer.TexturesVertex(textures)

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    renders = render_turntable_mesh(mesh, num_frames=30, image_size=image_size)
    imageio.mimsave("images/heart.gif", renders, fps=(30 // 3))

    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric",
                 "parametric_torus", "implicit", "implicit_torus", "heart"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size,
                              num_samples=args.num_samples)
    elif args.render == "parametric_torus":
        image = render_torus(image_size=args.image_size,
                             num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    elif args.render == "implicit_torus":
        image = render_torus_mesh(image_size=args.image_size)
    elif args.render == "heart":
        image = render_heart(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    plt.imsave(args.output_path, image)
