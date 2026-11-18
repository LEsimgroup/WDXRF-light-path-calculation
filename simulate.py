"""
xrf_sim/simulate.py
===================
纯模拟函数 ``simulate``：执行一次完整的 XRF 荧光模拟管线
（wafer → detector slit → 晶体 → 正比计数器窗口），不含任何绘图逻辑。

由 ``debug_tool`` 与 ``ui`` 共同复用，保证统计口径与可视化一致。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .core import WaferEmitter, BraggDiffraction
from .core.geometry import (
    bragg_angle_rad,
    detector_geometry,
)


@dataclass
class SimulationConfig:
    """一次完整模拟的全部可调参数。"""

    crystal_type: str = "flat"
    energy_eV: float = 183.0
    d_spacing: float = 8.95e-9
    detector_distance: float = 124.0
    detector_theta_deg: float = 30.0
    detector_size: Tuple[float, float] = (25.0, 5.0)
    crystal_distance: float = 30.0
    crystal_size: Tuple[float, float] = (30.0, 40.0)
    N_photons: int = 500_000
    fwhm_deg: float = 0.02
    spot_radius: float = 20.0
    is_flatted: bool = False
    ROC: Optional[float] = None
    a_crystal: Optional[float] = None
    spot_profile: Optional[tuple] = None
    pinhole_center: Tuple[float, float, float] = (0.0, 0.0, 14.02)
    pinhole_radius: float = 11.45
    pc_window_size: Tuple[float, float] = (25.0, 5.0)
    cone_angle_deg: Optional[float] = None
    seed: Optional[int] = None
    verbose: bool = True


@dataclass
class SimulationResult:
    """模拟管线各级结果。"""

    config: SimulationConfig
    bragg_angle_deg: float
    # 几何
    detector_normal: np.ndarray
    detector_center: np.ndarray
    crystal_normal: np.ndarray
    crystal_center: np.ndarray
    pc_center: np.ndarray
    pc_normal: np.ndarray
    pc_size: Tuple[float, float]
    # 光子计数
    N_emitted: int
    N_after_slit: int
    N_hit_crystal: int
    N_reflected: int
    N_pc_hit: int
    efficiency: float
    pc_efficiency: float
    incident_theta_mean: float
    # 光子数据
    photons_all: np.ndarray
    photons_slit: np.ndarray
    photons_reflected: np.ndarray          # 反射后（方向为镜面反射方向）
    photons_reflected_in: np.ndarray       # 反射光子的入射段（原始方向/起点）
    photons_pc: np.ndarray                 # 命中 PC 窗口的反射光子
    # PC 命中光子溯源
    pc_hit_slit_indices: np.ndarray        # PC 命中光子在 photons_slit 中的索引（与 pc_hit_points 一一对应）
    wafer_origins_pc: np.ndarray           # PC 命中光子的 wafer 发射原点 (x, y, 0)，shape (N_pc, 3)
    # 命中点
    hit_mask_slit: np.ndarray              # photons_slit 上几何命中晶体的掩码
    hit_points_crystal: np.ndarray         # 命中晶体的点
    reflected_hit_points: np.ndarray       # 被反射的命中点
    pc_hit_points: np.ndarray              # PC 窗口上的命中点
    pc_crystal_hit_points: np.ndarray      # 命中 PC 的反射光子在晶体上的命中点（与 pc_hit_points 一一对应）


def simulate(config: SimulationConfig) -> SimulationResult:
    """
    执行一次完整模拟。

    返回
    ----
    SimulationResult
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    import contextlib
    import io
    if not config.verbose:
        suppress = contextlib.redirect_stdout(io.StringIO())
    else:
        suppress = contextlib.nullcontext()

    with suppress:
        return _run(config)


def _run(cfg: SimulationConfig) -> SimulationResult:
    crystal_type = cfg.crystal_type.lower().strip()
    valid = ("flat", "cylindrical", "spiral")
    if crystal_type not in valid:
        raise ValueError(f"crystal_type 必须是 {valid} 之一，收到 '{crystal_type}'。")
    if crystal_type == "cylindrical" and cfg.ROC is None:
        raise ValueError("crystal_type='cylindrical' 时必须提供 ROC。")
    if crystal_type == "spiral" and cfg.a_crystal is None:
        raise ValueError("crystal_type='spiral' 时必须提供 a_crystal。")

    # 1. 布拉格角
    theta_B_rad = bragg_angle_rad(cfg.energy_eV, cfg.d_spacing)
    theta_B_deg = float(np.degrees(theta_B_rad))

    # 1b. 晶体到探测器中心距离
    # 柱面 Johann 弯晶：detector 到晶体中心距离 = ROC * sin(θ_B)（Johann 聚焦条件），
    # PC 到晶体中心距离与之等长（对称布拉格几何，由 pc_window_geometry 保证）。
    # 平晶 / 螺线晶体仍使用用户提供的 crystal_distance。
    if crystal_type == "cylindrical" and cfg.ROC is not None:
        crystal_distance = cfg.ROC * np.sin(theta_B_rad)
        print(f"[Simulate] cylindrical: crystal_distance = ROC*sin(θ_B) "
              f"= {cfg.ROC:.3f} * {np.sin(theta_B_rad):.4f} = {crystal_distance:.3f} mm")
    else:
        crystal_distance = cfg.crystal_distance

    # 2. 发射
    spectrum = lambda n: cfg.energy_eV * np.ones(n)
    emitter = WaferEmitter(
        N_photons=cfg.N_photons,
        spectrum_sampler=spectrum,
        spot_radius=cfg.spot_radius,
        spot_profile=cfg.spot_profile,
    )
    photons_slit, photons_all = emitter.emit(
        detector_distance=cfg.detector_distance,
        detector_theta_deg=cfg.detector_theta_deg,
        detector_size=cfg.detector_size,
        is_flatted=cfg.is_flatted,
        cone_angle_deg=cfg.cone_angle_deg,
    )

    # 3. BraggDiffraction
    bragg_kwargs = dict(
        d_spacing=cfg.d_spacing,
        crystal_theta_deg=theta_B_deg,
        fwhm_deg=cfg.fwhm_deg,
        pinhole_center=cfg.pinhole_center,
        pinhole_radius=cfg.pinhole_radius,
    )
    if crystal_type == "flat":
        bragg_kwargs["crystal_type"] = "flat"
    elif crystal_type == "cylindrical":
        bragg_kwargs["crystal_type"] = "cylindrical"
        bragg_kwargs["ROC"] = cfg.ROC
    elif crystal_type == "spiral":
        bragg_kwargs["crystal_type"] = "spiral"
        bragg_kwargs["detector_distance"] = cfg.detector_distance
        bragg_kwargs["detector_deg"] = cfg.detector_theta_deg
        bragg_kwargs["a_crystal"] = cfg.a_crystal
    bragg = BraggDiffraction(**bragg_kwargs)

    # 4. 探测器/晶体几何
    det_normal, det_center = detector_geometry(
        cfg.detector_distance, cfg.detector_theta_deg
    )
    # 晶体中心几何由各晶体模型自身提供（螺线晶体中心与平晶/弯晶不同）
    cryst_center, cryst_normal, _src_dist = bragg._crystal.crystal_frame(
        det_center, det_normal, crystal_distance
    )

    # 5. 布拉格反射 + PC 窗口
    pc = bragg.process(
        photons=photons_slit,
        _detector_normal=det_normal,
        _detector_center=det_center,
        _crystal_distance=crystal_distance,
        _crystal_size=cfg.crystal_size,
        return_pc=True,
        pc_window_size=cfg.pc_window_size,
    )

    N_emitted = cfg.N_photons
    N_after_slit = len(photons_slit)
    N_hit = pc["hit_crystal_N"]
    N_reflected = len(pc["reflected_photons"])
    N_pc = pc["pc_hit_count"]
    efficiency = N_reflected / N_emitted if N_emitted > 0 else 0.0
    pc_efficiency = N_pc / N_emitted if N_emitted > 0 else 0.0

    return SimulationResult(
        config=cfg,
        bragg_angle_deg=theta_B_deg,
        detector_normal=det_normal,
        detector_center=det_center,
        crystal_normal=cryst_normal,
        crystal_center=cryst_center,
        pc_center=pc["pc_center"],
        pc_normal=pc["pc_normal"],
        pc_size=pc["pc_size"],
        N_emitted=N_emitted,
        N_after_slit=N_after_slit,
        N_hit_crystal=N_hit,
        N_reflected=N_reflected,
        N_pc_hit=N_pc,
        efficiency=efficiency,
        pc_efficiency=pc_efficiency,
        incident_theta_mean=theta_B_deg,
        photons_all=photons_all,
        photons_slit=photons_slit,
        photons_reflected=pc["reflected_photons"],
        photons_reflected_in=pc["reflected_photons_in"],
        photons_pc=pc["pc_hit_photons"],
        pc_hit_slit_indices=pc["pc_hit_slit_indices"],
        wafer_origins_pc=(photons_slit["origin"][pc["pc_hit_slit_indices"]]
                          if N_pc > 0 else np.empty((0, 3))),
        hit_mask_slit=pc["hit_mask"],
        hit_points_crystal=pc["hit_points"],
        reflected_hit_points=pc["reflected_hit_points"],
        pc_hit_points=pc["pc_hit_points"],
        pc_crystal_hit_points=pc["pc_crystal_hit_points"],
    )


def pc_local_frame(pc_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造 PC 窗口的局部坐标系 (u, v, w)。

    - w = pc_normal（指向晶体）
    - u = 全局 x 方向投影到 PC 平面（若与法线近平行则退化为 y 方向）
    - v = cross(w, u)

    返回
    ----
    u, v, w : ndarray, shape (3,)，单位向量
    """
    w = pc_normal / np.linalg.norm(pc_normal)
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, w)) > 0.99:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, w) * w
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    v /= np.linalg.norm(v)
    return u, v, w


def save_pc_photons(result: SimulationResult, filepath: str) -> str:
    """
    将 PC 窗口接收到的所有光子保存为文本文件。

    每个光子记录：
        - 命中点位置在 PC 窗口局部坐标系下的分量 (u, v, w)（mm）
        - 方向向量在 PC 窗口局部坐标系下的分量 (du, dv, dw)（单位向量）
        - 能量 energy（eV）
        - 全局坐标位置 (x, y, z) 与方向 (dx, dy, dz)（便于核对）

    局部坐标系约定：
        w = pc_normal（指向晶体）
        u = 全局 x 投影到 PC 平面
        v = cross(w, u)

    返回：
        保存的文件路径。
    """
    import os
    import csv

    photons = result.photons_pc
    pc_center = np.asarray(result.pc_center, dtype=float)
    pc_normal = np.asarray(result.pc_normal, dtype=float)

    u, v, w = pc_local_frame(pc_normal)

    # 命中点（PC 窗口上的实际位置）
    hit_points = result.pc_hit_points
    if len(hit_points) == 0:
        hit_points = np.asarray(photons["origin"]).reshape(-1, 3)
    hit_points = np.asarray(hit_points, dtype=float).reshape(-1, 3)

    dirs = np.asarray(photons["direction"]).reshape(-1, 3)
    energies = np.asarray(photons["energy"]).reshape(-1)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "u_mm", "v_mm", "w_mm",
            "du", "dv", "dw",
            "x_mm", "y_mm", "z_mm",
            "dx", "dy", "dz",
            "energy_eV",
        ])
        for i in range(len(hit_points)):
            pos = hit_points[i]
            d = dirs[i]
            # 位置相对 PC 中心的 uv 分量
            rel = pos - pc_center
            pu = float(np.dot(rel, u))
            pv = float(np.dot(rel, v))
            pw = float(np.dot(rel, w))
            # 方向在 uv 坐标系下的分量
            du = float(np.dot(d, u))
            dv = float(np.dot(d, v))
            dw = float(np.dot(d, w))
            writer.writerow([
                f"{pu:.6f}", f"{pv:.6f}", f"{pw:.6f}",
                f"{du:.6f}", f"{dv:.6f}", f"{dw:.6f}",
                f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                f"{d[0]:.6f}", f"{d[1]:.6f}", f"{d[2]:.6f}",
                f"{energies[i]:.6f}",
            ])

    print(f"[save_pc_photons] Saved {len(hit_points)} photons to {filepath}")
    return filepath


def _region_point_mask(x, y, region) -> np.ndarray:
    """
    对任意点坐标数组 (x, y) 返回是否落在 region 内的布尔掩码。

    region 支持两种形式：
        {"type": "rect",   "x_range": (x0, x1), "y_range": (y0, y1)}
        {"type": "circle", "center": (cx, cy), "radius": r}
    """
    rtype = str(region.get("type", "rect")).lower()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if rtype == "rect":
        x0, x1 = sorted(region["x_range"])
        y0, y1 = sorted(region["y_range"])
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if rtype == "circle":
        cx, cy = region["center"]
        r = float(region["radius"])
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    raise ValueError(f"未知的 region 类型 '{rtype}'，可选 'rect' 或 'circle'。")


def evaluate_regions(result: SimulationResult, regions):
    """
    按用户指定的区域（矩形 / 圆形）统计 PC 接收光强的占比。

    对每个区域 R 统计以下指标：

        - ``pc_fraction``  ← **核心指标**
              该区域内 PC 接收到的光强占「全部 PC 接收光强」的比例：
                  N_pc(R) / N_pc_total
              若各区域互斥且铺满整个发光区，所有区域之和 = 100%。
              （``pc_share`` 为同一指标的别名，保留向后兼容。）
        - ``pc_intensity_fraction``
              该区域 PC 接收到的光强占「全部发射光强」的比例：
                  N_pc(R) / N_emit_total
              用于判断该区域对总光强的贡献（互斥铺满时之和 ≈ 总 PC 效率）。
        - ``intensity_fraction``
              该区域占 wafer profile 总强度的比例：I(R) / I_total。
        - ``collection_efficiency``
              该区域自身的收集效率：N_pc(R) / N_emit(R)。

    参数
    ----
    result : SimulationResult
        一次模拟的结果（需含 photons_all 与 wafer_origins_pc）。
    regions : list of dict
        区域列表，每个元素形如::

            {"type": "rect",   "x_range": (x0, x1), "y_range": (y0, y1),
             "label": "中心区"}
            {"type": "circle", "center": (cx, cy), "radius": r,
             "label": "内圈"}

    返回
    ----
    stats : list of dict
        与 ``regions`` 一一对应的统计结果列表。
    summary : dict
        全局参考量：总发射数、总 PC 命中数、总 PC 效率、profile 总强度等。
    """
    origins_all = np.asarray(result.photons_all["origin"], dtype=float).reshape(-1, 3)
    pc_origins = np.asarray(result.wafer_origins_pc, dtype=float).reshape(-1, 3)

    n_emit_total = len(origins_all)
    n_pc_total = len(pc_origins)

    # profile 网格（用于计算强度占比）
    profile = result.config.spot_profile
    if profile is not None:
        prof = np.asarray(profile[0], dtype=float)
        xmin, xmax = profile[1]
        ymin, ymax = profile[2]
        Ny, Nx = prof.shape
        total_intensity = float(prof.sum())
        dx = (xmax - xmin) / Nx
        dy = (ymax - ymin) / Ny
        xc = xmin + (np.arange(Nx) + 0.5) * dx
        yc = ymin + (np.arange(Ny) + 0.5) * dy
        Xc, Yc = np.meshgrid(xc, yc)   # shape (Ny, Nx)
    else:
        prof = None
        total_intensity = None
        Xc = Yc = None

    stats = []
    for i, region in enumerate(regions):
        label = region.get("label")
        if not label:
            rtype = str(region.get("type", "rect")).lower()
            if rtype == "circle":
                cx, cy = region["center"]
                label = f"circle{i + 1}({cx:g},{cy:g},r={region['radius']:g})"
            else:
                x0, x1 = sorted(region["x_range"])
                y0, y1 = sorted(region["y_range"])
                label = f"rect{i + 1}[{x0:g},{x1:g}]x[{y0:g},{y1:g}]"

        # 区域内的发射光子数 / PC 命中数
        m_emit = _region_point_mask(origins_all[:, 0], origins_all[:, 1], region)
        n_emit_R = int(m_emit.sum())
        m_pc = _region_point_mask(pc_origins[:, 0], pc_origins[:, 1], region)
        n_pc_R = int(m_pc.sum())

        # 区域内的 profile 强度
        if prof is not None:
            cell_mask = _region_point_mask(Xc, Yc, region)
            intensity_R = float(prof[cell_mask].sum())
            intensity_fraction = (intensity_R / total_intensity
                                  if total_intensity > 0 else None)
        else:
            intensity_R = None
            intensity_fraction = None

        pc_fraction = (n_pc_R / n_pc_total if n_pc_total > 0 else None)
        stats.append({
            "label": label,
            "type": str(region.get("type", "rect")).lower(),
            "region": region,
            "n_emit_in_region": n_emit_R,
            "n_pc_in_region": n_pc_R,
            "intensity": intensity_R,
            # 核心指标：区域内 PC 光强 / 全部 PC 光强
            "pc_fraction": pc_fraction,
            "pc_share": pc_fraction,          # 别名，向后兼容
            "pc_intensity_fraction": (n_pc_R / n_emit_total
                                      if n_emit_total > 0 else None),
            "intensity_fraction": intensity_fraction,
            "collection_efficiency": (n_pc_R / n_emit_R if n_emit_R > 0 else None),
        })

    summary = {
        "n_emit_total": n_emit_total,
        "n_pc_total": n_pc_total,
        "pc_efficiency": (n_pc_total / n_emit_total
                          if n_emit_total > 0 else None),
        "total_intensity": total_intensity,
        "has_profile": prof is not None,
        "pc_fraction_sum": sum(s["n_pc_in_region"] for s in stats) / n_pc_total
                           if n_pc_total > 0 else None,
    }
    return stats, summary
