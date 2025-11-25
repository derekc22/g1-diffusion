import numpy as np
import torch
import torch.nn.functional as F


def quat_wxyz_to_xyzw(quat):
    """
    Input: (..., 4) quaternion in wxyz order.
    Output: (..., 4) quaternion in xyzw order.
    """
    if isinstance(quat, np.ndarray):
        w, x, y, z = np.split(quat, 4, axis=-1)
        return np.concatenate([x, y, z, w], axis=-1)
    elif isinstance(quat, torch.Tensor):
        w = quat[..., 0:1]
        x = quat[..., 1:2]
        y = quat[..., 2:3]
        z = quat[..., 3:4]
        return torch.cat([x, y, z, w], dim=-1)
    else:
        raise TypeError(f"Unsupported type for quat_wxyz_to_xyzw: {type(quat)}")


def quat_to_rot6d_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (xyzw) to 6D rotation representation (B, 6).

    Based on:
    - Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    """
    # normalize
    quat_xyzw = quat_xyzw / (quat_xyzw.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = torch.unbind(quat_xyzw, dim=-1)

    # rotation matrix elements
    xx = 1 - 2 * (y * y + z * z)
    xy = 2 * (x * y - z * w)
    xz = 2 * (x * z + y * w)
    yx = 2 * (x * y + z * w)
    yy = 1 - 2 * (x * x + z * z)
    yz = 2 * (y * z - x * w)

    # first two columns of rotation matrix
    r1 = torch.stack([xx, xy, xz], dim=-1)  # (..., 3)
    r2 = torch.stack([yx, yy, yz], dim=-1)  # (..., 3)

    rot6d = torch.cat([r1, r2], dim=-1)     # (..., 6)
    return rot6d


def rot6d_to_mat(r6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.

    r6: (..., 6)
    returns: (..., 3, 3)
    """
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    a2_ortho = a2 - dot * b1
    b2 = F.normalize(a2_ortho, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def mat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (xyzw).

    R: (..., 3, 3)
    returns: (..., 4) xyzw
    """
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]

    trace = m00 + m11 + m22

    qw = torch.empty_like(trace)
    qx = torch.empty_like(trace)
    qy = torch.empty_like(trace)
    qz = torch.empty_like(trace)

    mask = trace > 0
    if mask.any():
        t = torch.sqrt(trace[mask] + 1.0) * 2.0
        qw_mask = 0.25 * t
        qx_mask = (R[mask, 2, 1] - R[mask, 1, 2]) / t
        qy_mask = (R[mask, 0, 2] - R[mask, 2, 0]) / t
        qz_mask = (R[mask, 1, 0] - R[mask, 0, 1]) / t
        qw[mask], qx[mask], qy[mask], qz[mask] = qw_mask, qx_mask, qy_mask, qz_mask

    mask1 = (~mask) & (m00 >= m11) & (m00 >= m22)
    if mask1.any():
        t = torch.sqrt(1.0 + m00[mask1] - m11[mask1] - m22[mask1]) * 2.0
        qw_mask = (R[mask1, 2, 1] - R[mask1, 1, 2]) / t
        qx_mask = 0.25 * t
        qy_mask = (R[mask1, 0, 1] + R[mask1, 1, 0]) / t
        qz_mask = (R[mask1, 0, 2] + R[mask1, 2, 0]) / t
        qw[mask1], qx[mask1], qy[mask1], qz[mask1] = qw_mask, qx_mask, qy_mask, qz_mask

    mask2 = (~mask) & (~mask1) & (m11 >= m22)
    if mask2.any():
        t = torch.sqrt(1.0 + m11[mask2] - m00[mask2] - m22[mask2]) * 2.0
        qw_mask = (R[mask2, 0, 2] - R[mask2, 2, 0]) / t
        qx_mask = (R[mask2, 0, 1] + R[mask2, 1, 0]) / t
        qy_mask = 0.25 * t
        qz_mask = (R[mask2, 1, 2] + R[mask2, 2, 1]) / t
        qw[mask2], qx[mask2], qy[mask2], qz[mask2] = qw_mask, qx_mask, qy_mask, qz_mask

    mask3 = (~mask) & (~mask1) & (~mask2)
    if mask3.any():
        t = torch.sqrt(1.0 + m22[mask3] - m00[mask3] - m11[mask3]) * 2.0
        qw_mask = (R[mask3, 1, 0] - R[mask3, 0, 1]) / t
        qx_mask = (R[mask3, 0, 2] + R[mask3, 2, 0]) / t
        qy_mask = (R[mask3, 1, 2] + R[mask3, 2, 1]) / t
        qz_mask = 0.25 * t
        qw[mask3], qx[mask3], qy[mask3], qz[mask3] = qw_mask, qx_mask, qy_mask, qz_mask

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    return quat


def rot6d_to_quat_wxyz(r6: torch.Tensor) -> torch.Tensor:
    """
    r6: (..., 6)
    returns: (..., 4) quaternion in wxyz order
    """
    R = rot6d_to_mat(r6)
    quat_xyzw = mat_to_quat_xyzw(R)
    x = quat_xyzw[..., 0:1]
    y = quat_xyzw[..., 1:2]
    z = quat_xyzw[..., 2:3]
    w = quat_xyzw[..., 3:4]
    quat_wxyz = torch.cat([w, x, y, z], dim=-1)
    return quat_wxyz
