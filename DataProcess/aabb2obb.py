import numpy as np
import cv2 
from PIL import Image

def compute_2d_obb(points: np.ndarray):
    """
    输入:
        points: (N, 2) 二维点集，已投影到图像平面
    返回:
        box_pts: (4, 2) OBB 四个角点，按顺时针或逆时针顺序
        angle: 旋转角度（弧度），表示 OBB 相对图像 x 轴的旋转
    """
    # 1. 计算中心与协方差
    mean = np.mean(points, axis=0)
    cov = np.cov(points - mean, rowvar=False)

    # 2. PCA：特征向量与特征值
    eigvals, eigvecs = np.linalg.eigh(cov)  # 顺序为升序
    order = np.argsort(eigvals)[::-1]        # 降序索引
    principal_axis = eigvecs[:, order[0]]    # 主方向

    # 3. 计算旋转角度
    angle = np.arctan2(principal_axis[1], principal_axis[0])

    # 4. 构造旋转矩阵
    R_align = np.array([
    [ np.cos(-angle), -np.sin(-angle)],
    [ np.sin(-angle),  np.cos(-angle)]
    ])
    rot_pts = (points - mean) @ R_align

    # 6. 计算最小外接矩形在旋转坐标系下的 min/max
    min_xy = np.min(rot_pts, axis=0)
    max_xy = np.max(rot_pts, axis=0)

    # 7. 构造矩形四角（逆时针）
    box = np.array([
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
    ])

    # 8. invert with +angle:
    R_back = np.array([
        [ np.cos(angle), -np.sin(angle)],
        [ np.sin(angle),  np.cos(angle)]
    ])
    box_pts = (box @ R_back) + mean
    return box_pts, angle

def create_obb_mask( box_pts, image_shape):
    """
    在给定图像尺寸下，依据 OBB 四角坐标生成二值掩码。

    Args:
        image_shape: (height, width) 或 (height, width, channels) 的形状元组
        box_pts: (4, 2) 数组，表示 OBB 四个顶点 (x, y)，浮点或整数皆可

    Returns:
        mask: np.ndarray of shape (height, width), dtype=np.uint8，
              区域内为 1，区域外为 0
    """
    # 解析高度、宽度
    h, w = image_shape[:2]
    # 创建全 0 掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    # box_pts 需要是整数像素坐标
    # poly = np.array([box_pts], dtype=np.int32)  # shape=(1,4,2).reshape((-1, 1, 2))

    poly_pts = box_pts[:, [1, 0]]    # now [[y, x], ...]
    poly = np.array([poly_pts], dtype=np.int32)  # shape (1, 4, 2)

    cv2.fillPoly(mask, poly, 1)
    return mask