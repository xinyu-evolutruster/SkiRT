import torch
import pytorch3d

from utils import get_device, get_mesh_renderer, get_points_renderer


def render_turntable_mesh(
    mesh=None,
    num_frames=10,
    image_size=512,
    device=None
):

    if mesh == None:
        return

    if device == None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    # Place a point light in front of the cow.s
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0, -3]], device=device)

    renders = []
    for deg in range(0, 360, 360 // num_frames):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=3, elev=0, azim=deg + 180, degrees=True,
        )
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        renders.append(rend)

    return renders


def render_turntable_pcl(
    pcl=None,
    num_frames=10,
    image_size=256,
    dist=6,
    up=((0, 1, 0), ),
    rotate_dir=True,  # clockwise
    device=None
):
    if pcl == None:
        return None

    if device == None:
        device = get_device()
        # device = 'cpu'

    # Get the renderer.
    renderer = get_points_renderer(
        image_size=image_size,
        radius=0.01
    )

    renders = []

    for deg in range(0, 360, 360 // num_frames):
        azim = deg
        if rotate_dir == False:  # counter-clockwise
            azim = -azim

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=dist, elev=0, azim=azim, degrees=True, up=up
        )

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(pcl, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        renders.append(rend)

    return renders
