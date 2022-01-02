import torch
import numpy as np
import pdb 
import logging
from utils.geom_utils import orthographic_proj_withz, quat_rotate

logger_py = logging.getLogger(__name__)


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None, invert_y_axis=False):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and       # False -> pass!
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    if invert_y_axis:       # False -> pass!
        assert(image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def transform_to_world(pixels, depth, sfm_pose, scale_mat=None,
                       invert=True, use_absolute_depth=True):
    # pixel_world, camera_world 둘다 여기서 camera mat을 스쳐지나감!
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)
    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    # cam = to_pytorch(cam)
    device = pixels.device
    # 오케이 여기서부터 pixel 3차원으로 들어감!
    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)        # 이렇게 돌려주는 이유: 밑에 matrix 연산을 가능하게 하려고!
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

            # 16, 256, 3
    # Project pixels into camera space
    if use_absolute_depth:      # True  # depth = torch.ones(batch_size, n_pts, 1).to(device)
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # 선택지 1. 2D->3D
    # p_world = inverse_orthographic_proj_withz(pixels, cam)
    # 선택지 2. 3D->2D
    pixels = pixels.permute(0, 2, 1)[:, :, :-1]        # 이렇게 돌려주는 이유: 밑에 matrix 연산을 가능하게 하려고!

    p_world = orthographic_proj_withz(pixels, sfm_pose)

    # Transform p_world back to 3D coordinates
    # p_world = p_world.permute(0, 2, 1)       # 다 동일한 값으로 origins 존재! -> 계산 편의를 위해 permute!

    if is_numpy:
        p_world = p_world.numpy()   # (16, 256, 3)이어야 함
    return p_world


def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    p_world = torch.cat([p_world, torch.ones(
        batch_size, n_p, 1).to(device)], dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    p_cam = camera_mat @ world_mat @ scale_mat @ p_world

    # Transform points back to 3D coordinates
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def origin_to_world(n_points, sfm_pose, scale_mat=None,
                    invert=False):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: false)
    '''
    batch_size = sfm_pose.shape[0]
    device = sfm_pose.device
    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    # p[:, -1] = 1.       # (0, 0, 0, 1)에서부터 rotate       # -> camera mat, world mat의 translation에 따라 camera origin이 바뀜 

    '''
    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(     # take this value
            0).repeat(batch_size, 1, 1).to(device)

    # Invert matrices
    if invert:      # False                         -> 아 우리는 여기서 True로 넣어줘야 한다 + 좌표계도 변환해주어야 한다 
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation  # origin mat -> camera mat -> world mat 
    '''
    p = p.permute(0, 2, 1)[:, :, :-1]
    p_world = orthographic_proj_withz(p, sfm_pose)
    # p_world = scale_mat @ world_mat @ camera_mat @ p     # (batch, 4, resxres)   # scale mat: identity   
    # Transform points back to 3D coordinates

    # p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


def image_points_to_world(image_points, sfm_pose, scale_mat=None,
                          invert=False, negative_depth=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: False)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    if negative_depth:
        d_image *= -1.
    return transform_to_world(image_points, d_image, sfm_pose,
                              scale_mat, invert=invert)


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z


def orthographic_proj_withz(x3d, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # 1. quaternion rotation 
    X_rot = quat_rotate(x3d, quat)

    # 2. multiply by scale
    proj = scale * X_rot

    # 3. plus trans
    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)


# newly added by mira 
def inverse_orthographic_proj_withz(x2d, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)


    # 오키.. 1.2.는 완전 orthographic projection용도..!
    # 1. minus trans 
    x2d[:, :, :2] = x2d[:, :, :2] - trans

    # 2. divide by scale 
    x2d = x2d / scale

    # 3. inverese quaternion rotation 
    rot = quat_to_rotmat(quat)
    dumm = torch.tensor([0, 0, 0]).reshape(1, 3, 1).repeat(batch_size, 1, 1)
    inverse_rot = torch.cat([torch.inverse(rot), dumm], dim=-1)
    inverse_quat = tgm.rotation_matrix_to_quaternion(inverse_rot)   
    x2d = quat_rotate(x2d, inverse_quat)

    return x2d      # 3d points     cmr에서는 이 다음에 뭘하는거지..? 
