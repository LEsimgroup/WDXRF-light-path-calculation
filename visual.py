"""
utils/visual.py
===============
XRF 荧光模拟的可视化工具函数。

本模块仅包含纯粹的绘图函数，不包含任何模拟逻辑或实验脚本。
所有几何计算均通过调用 ``core.geometry`` 中的工具函数完成。

提供的函数
----------
- plot_photon_paths_all      : 可视化全部光子路径，区分命中与未命中
- plot_photon_paths          : 仅可视化命中探测器的光子路径
- plot_detector_to_crystal   : 可视化 wafer → 探测器 → 晶体 的完整路径
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401（注册 3D 投影）


# ---------------------------------------------------------------------------
# 内部绘图辅助函数
# ---------------------------------------------------------------------------

def _draw_rect_plane(ax, center: np.ndarray, normal: np.ndarray,
                     size: tuple, color: str = "k", label: str = None):
    """在 3D 坐标轴上绘制一个矩形平面的轮廓线。"""
    width, height = size
    u = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, normal)) > 0.99:
        u = np.array([0.0, 1.0, 0.0])
    u -= np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    corners = np.array([
        center + s1 * (width / 2) * u + s2 * (height / 2) * v
        for s1, s2 in [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            color=color, label=label)


def _draw_cylindrical_crystal(ax, crystal_center: np.ndarray,
                               detector_normal: np.ndarray,
                               crystal_theta_deg: float,
                               detector_deg: float,
                               crystal_size: tuple,
                               ROC: float,
                               color: str = "b"):
    """
    在 3D 坐标轴上绘制约翰型弯晶的圆弧曲面。

    弯晶在 yz 平面内弯曲，绕 x 轴旋转到对应角度后平移至晶体中心。
    """
    from xrf_sim.core.geometry import rotation_x

    width, arc_length = crystal_size
    phi_range = arc_length / ROC / 2
    n_arc = 100

    phi = np.linspace(-phi_range, phi_range, n_arc)
    y_arc = ROC * np.sin(phi)
    z_arc = ROC * (-1.0 + np.cos(phi))
    x_vals = np.linspace(-width / 2, width / 2, 10)

    X, Y = np.meshgrid(x_vals, y_arc)
    Z = np.tile(z_arc, (len(x_vals), 1)).T

    # 旋转到晶体坐标系
    theta_total = np.radians(180.0 - crystal_theta_deg - detector_deg)
    R = rotation_x(theta_total)
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    rotated = (R @ pts.T).T + crystal_center

    Xr = rotated[:, 0].reshape(Y.shape)
    Yr = rotated[:, 1].reshape(Y.shape)
    Zr = rotated[:, 2].reshape(Y.shape)
    ax.plot_surface(Xr, Yr, Zr, color=color, alpha=0.3)


# ---------------------------------------------------------------------------
# 公开绘图函数
# ---------------------------------------------------------------------------

def plot_photon_paths_all(photons,
                          photons_all,
                          detector_center: np.ndarray,
                          detector_normal: np.ndarray,
                          detector_size: tuple,
                          max_rays: int = 500):
    """
    可视化全部光子路径，命中探测器的光子用红色标出，未命中的用灰色。

    参数
    ----
    photons : ndarray of PHOTON_DTYPE
        命中探测器的光子。
    photons_all : ndarray of PHOTON_DTYPE
        全部发射的光子。
    detector_center : ndarray, shape (3,)
    detector_normal : ndarray, shape (3,)
    detector_size : tuple of float
        (width, height)，mm。
    max_rays : int
        最多绘制的光线数量。
    """
    from scipy.spatial import cKDTree

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # 探测器轮廓
    _draw_rect_plane(ax, detector_center, detector_normal, detector_size,
                     color="k", label="Detector")

    # wafer 面
    wg = np.linspace(-2, 2, 2)
    ax.plot_surface(*np.meshgrid(wg, wg), np.zeros((2, 2)),
                    alpha=0.2, color="gray", edgecolor="none")

    # 标记命中光子
    is_hit = np.zeros(len(photons_all), dtype=bool)
    if len(photons) > 0:
        tree_o = cKDTree(photons_all["origin"])
        tree_d = cKDTree(photons_all["direction"])
        idx_o  = tree_o.query(photons["origin"],  k=1)[1]
        idx_d  = tree_d.query(photons["direction"], k=1)[1]
        is_hit[np.intersect1d(idx_o, idx_d)] = True

    idx_all = np.arange(len(photons_all))
    np.random.shuffle(idx_all)
    for i in idx_all[:max_rays]:
        o = photons_all["origin"][i]
        d = photons_all["direction"][i]
        denom = np.dot(d, detector_normal)
        if np.abs(denom) < 1e-6:
            continue
        t = np.dot(detector_center - o, detector_normal) / denom
        if t <= 0:
            continue
        hit = o + t * d
        color = "r" if is_hit[i] else "gray"
        alpha = 0.5 if is_hit[i] else 0.2
        ax.plot([o[0], hit[0]], [o[1], hit[1]], [o[2], hit[2]],
                color=color, alpha=alpha)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Photon paths (red = hit detector)")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.show()


def plot_photon_paths(photons,
                      detector_center: np.ndarray,
                      detector_normal: np.ndarray,
                      detector_size: tuple,
                      max_rays: int = 500):
    """
    仅可视化命中探测器的光子路径（wafer → 探测器）。

    参数
    ----
    photons : ndarray of PHOTON_DTYPE
        命中探测器的光子。
    detector_center : ndarray, shape (3,)
    detector_normal : ndarray, shape (3,)
    detector_size : tuple of float
    max_rays : int
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    _draw_rect_plane(ax, detector_center, detector_normal, detector_size, color="k")

    wg = np.linspace(-2, 2, 2)
    ax.plot_surface(*np.meshgrid(wg, wg), np.zeros((2, 2)),
                    alpha=0.2, color="gray", edgecolor="none")

    n_show = min(max_rays, len(photons))
    idx    = np.random.choice(len(photons), size=n_show, replace=False)
    for i in idx:
        o = photons["origin"][i]
        d = photons["direction"][i]
        denom = np.dot(d, detector_normal)
        if np.abs(denom) < 1e-6:
            continue
        t = np.dot(detector_center - o, detector_normal) / denom
        hit = o + t * d
        ax.plot([o[0], hit[0]], [o[1], hit[1]], [o[2], hit[2]], "r-", alpha=0.4)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Photon paths hitting the detector")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.show()


def plot_full_path(result, max_rays: int = 300):
    """
    可视化完整光路 wafer → detector slit → 晶体 → 正比计数器（PC）窗口。

    参数
    ----
    result : SimulationResult
        由 ``xrf_sim.simulate`` 返回的结果对象。
    max_rays : int
        最多绘制的光线数量。
    """
    from ..plotting import draw_3d, draw_yz

    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    draw_3d(ax3d, result, max_rays=max_rays)
    ax_yz = fig.add_subplot(1, 2, 2)
    draw_yz(ax_yz, result, max_rays=max_rays)
    fig.tight_layout()
    plt.show()


def plot_detector_to_crystal(photons_reflected,
                              detector_center: np.ndarray,
                              detector_normal: np.ndarray,
                              detector_size: tuple,
                              crystal_center: np.ndarray,
                              crystal_normal: np.ndarray,
                              crystal_size: tuple,
                              crystal_theta_deg: float = None,
                              detector_deg: float = None,
                              crystal_ROC=None,
                              max_rays: int = 300):
    """
    可视化 wafer → 探测器 → 晶体 的完整光子路径。

    支持平晶（crystal_ROC=None）和约翰型弯晶（crystal_ROC 为数值）两种情况。

    参数
    ----
    photons_reflected : ndarray of PHOTON_DTYPE
        被晶体反射的光子。
    detector_center, detector_normal, detector_size
        探测器几何参数。
    crystal_center, crystal_normal, crystal_size
        晶体几何参数。
    crystal_theta_deg : float, optional
        晶体布拉格角（度），绘制弯晶时需要。
    detector_deg : float, optional
        探测器掠射角（度），绘制弯晶时需要。
    crystal_ROC : float or None
        弯晶曲率半径（mm）；为 None 时视为平晶。
    max_rays : int
        最多绘制的光线数量。
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # 探测器
    _draw_rect_plane(ax, detector_center, detector_normal, detector_size,
                     color="k", label="Detector")

    # 晶体
    if crystal_ROC is None:
        _draw_rect_plane(ax, crystal_center, crystal_normal, crystal_size,
                         color="b", label="Flat Crystal")
    else:
        _draw_cylindrical_crystal(
            ax, crystal_center, detector_normal,
            crystal_theta_deg, detector_deg, crystal_size, crystal_ROC,
            color="b"
        )

    # 光线路径
    n_show = min(max_rays, len(photons_reflected))
    if n_show == 0:
        print("[Visual] 无反射光子可绘制。")
    else:
        idx = np.random.choice(len(photons_reflected), size=n_show, replace=False)
        print(f"[Visual] 最终被反射的光子数：{len(photons_reflected)}")

        for i in idx:
            o = photons_reflected["origin"][i]
            d = photons_reflected["direction"][i]

            # wafer → 探测器
            denom1 = np.dot(d, detector_normal)
            if np.abs(denom1) < 1e-6:
                continue
            t1   = np.dot(detector_center - o, detector_normal) / denom1
            hit1 = o + t1 * d

            # 探测器 → 晶体（沿原始方向延伸到晶体法线平面）
            denom2 = np.dot(d, crystal_normal)
            if np.abs(denom2) < 1e-6:
                continue
            t2 = np.dot(crystal_center - o, crystal_normal) / denom2
            if t2 <= 0:
                continue
            hit2 = o + t2 * d

            ax.plot([o[0], hit1[0]], [o[1], hit1[1]], [o[2], hit1[2]],
                    "r-", alpha=0.5)
            ax.plot([hit1[0], hit2[0]], [hit1[1], hit2[1]], [hit1[2], hit2[2]],
                    "b-", alpha=0.6)

    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 250)
    ax.set_zlim(0, 200)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Photon Paths: Wafer → Detector → Crystal")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.show()
