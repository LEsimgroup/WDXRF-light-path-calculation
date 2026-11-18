"""
xrf_sim/plotting.py
===================
由 ``SimulationResult`` 生成的可复用绘图函数。

提供：
- ``build_figures(result, max_vis_rays)`` — 一次生成 overview / 3D / YZ 三张图（debug_tool 用）。
- ``draw_3d(ax, result, max_rays)``        — 在已有 3D axes 上绘制完整光路（UI 用）。
- ``draw_yz(ax, result, max_rays)``        — 在已有 2D axes 上绘制 YZ 投影（UI 用）。

光路：wafer → detector slit → 晶体 → 正比计数器（PC）窗口。
反射段遵循镜面反射定律，由 simulate() 计算并随结果传入。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .core.geometry import rotation_x


def _setup_cjk_font_fallback():
    """
    为中文字符配置文件级字体回退。

    matplotlib 3.6+ 支持逐字符字体回退，但要求把候选字体写进
    ``font.family`` 列表（仅改 ``font.sans-serif`` 不触发回退）。
    这里把可用的中文字体**追加**在 ``sans-serif`` 之后，于是英文仍由
    DejaVu Sans 渲染、中文自动回退到中文字体，不改变原有图表外观。
    """
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
        "Source Han Sans SC", "PingFang SC", "WenQuanYi Micro Hei",
        "Arial Unicode MS", "DengXian",
    ]
    # 只启用第一个可用的中文字体：多个中文字体同时作为回退候选时，
    # matplotlib 会逐个尝试并打印 findfont 权重缺失噪声。
    cjk = next((name for name in candidates if name in available), None)
    if cjk is None:
        return
    family = list(plt.rcParams.get("font.family", []))
    if cjk not in family:
        family.append(cjk)
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


_setup_cjk_font_fallback()


_setup_cjk_font_fallback()


# ---------------------------------------------------------------------------
# 基础绘图辅助
# ---------------------------------------------------------------------------

def _rect_basis(normal):
    """给定法线，返回矩形平面内的两个正交单位向量 (u, v)。"""
    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, normal)) > 0.99:
        u = np.array([0.0, 1.0, 0.0])
    u = u - np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def draw_rect_3d(ax, center, normal, size, color="k", linewidth=2, label=None):
    """在 3D axes 上绘制矩形平面轮廓。"""
    width, height = size
    u, v = _rect_basis(normal)
    corners = np.array([
        center + s1 * (width / 2) * u + s2 * (height / 2) * v
        for s1, s2 in [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            color=color, linewidth=linewidth, label=label)


def _crystal_yz_curve(result):
    """返回晶体在 YZ 平面的轮廓点 (y, z)，根据晶体类型绘制。"""
    cfg = result.config
    cc = result.crystal_center
    if cfg.crystal_type == "flat":
        n = result.crystal_normal[[1, 2]]
        n = n / np.linalg.norm(n)
        # 沿与法线垂直的方向取半长
        t = np.array([-n[1], n[0]])
        hw = cfg.crystal_size[1] / 2
        return np.array([cc[1] + hw * t[0], cc[1] - hw * t[0]]), \
               np.array([cc[2] + hw * t[1], cc[2] - hw * t[1]])
    if cfg.crystal_type == "cylindrical" and cfg.ROC is not None:
        arc_len = cfg.crystal_size[1]
        phi_range = arc_len / cfg.ROC / 2
        phi = np.linspace(-phi_range, phi_range, 200)
        y_arc = cfg.ROC * np.sin(phi)
        z_arc = cfg.ROC * (-1.0 + np.cos(phi))
        theta_total = np.radians(180.0 - result.bragg_angle_deg
                                 - cfg.detector_theta_deg)
        R = rotation_x(theta_total)
        pts = np.stack([np.zeros_like(y_arc), y_arc, z_arc], axis=1)
        rot = (R @ pts.T).T + cc
        return rot[:, 1], rot[:, 2]
    # spiral
    a = cfg.a_crystal
    b = 1.0 / np.tan(np.radians(result.bragg_angle_deg))
    S = result.detector_center
    away = S[[1, 2]]  # wafer 在原点
    phi0 = np.arctan2(away[1], away[0])
    phi = np.linspace(phi0 - np.radians(30), phi0 + np.radians(30), 1000)
    r = a * np.exp(b * (phi - phi0))
    return S[1] + r * np.cos(phi), S[2] + r * np.sin(phi)


# ---------------------------------------------------------------------------
# 3D 光路图
# ---------------------------------------------------------------------------

def draw_3d(ax, result, max_rays=300):
    """在 3D axes 上绘制 wafer→slit→晶体→PC 完整光路。"""
    cfg = result.config
    det_c, det_n = result.detector_center, result.detector_normal
    cryst_c, cryst_n = result.crystal_center, result.crystal_normal
    pc_c, pc_n = result.pc_center, result.pc_normal

    ax.clear()
    # Wafer
    wg = np.linspace(-30, 30, 2)
    Xw, Yw = np.meshgrid(wg, wg)
    ax.plot_surface(Xw, Yw, np.zeros_like(Xw),
                    alpha=0.15, color="gray", edgecolor="none")
    ax.text(0, 0, -3, "Wafer", fontsize=9, color="black", ha="center")

    # 探测器 slit
    draw_rect_3d(ax, det_c, det_n, cfg.detector_size,
                 color="green", label="Detector Slit")
    # 晶体（矩形近似）
    draw_rect_3d(ax, cryst_c, cryst_n, cfg.crystal_size,
                 color="blue", label="Crystal")
    # PC 窗口
    draw_rect_3d(ax, pc_c, pc_n, cfg.pc_window_size,
                 color="purple", label="PC Window")

    photons = result.photons_slit
    N_after_slit = len(photons)
    hit_pts = result.hit_points_crystal
    hit_mask_slit = result.hit_mask_slit

    # 抽样：命中晶体（红，画到晶体命中点）与未命中（灰，画到 slit）
    n_hit = int(hit_mask_slit.sum())
    n_show_hit = min(max_rays, n_hit)
    n_show_miss = min(max(0, max_rays - n_show_hit), N_after_slit - n_hit)

    hit_slit_idx = np.where(hit_mask_slit)[0]
    chosen_hit = (np.random.choice(hit_slit_idx, size=n_show_hit, replace=False)
                  if n_show_hit else np.array([], dtype=int))
    miss_idx = np.setdiff1d(np.arange(N_after_slit), hit_slit_idx)
    chosen_miss = (np.random.choice(miss_idx, size=n_show_miss, replace=False)
                   if n_show_miss and len(miss_idx) else np.array([], dtype=int))

    # hit_pts 与 hit_slit_idx 按位置一一对应（intersect 按 hit_mask True 顺序返回）
    hit_pt_for_slit = {int(hit_slit_idx[i]): hit_pts[i]
                       for i in range(len(hit_slit_idx))} if len(hit_pts) else {}

    for i in chosen_hit:
        o = photons["origin"][i]
        end = hit_pt_for_slit.get(int(i))
        if end is None:
            continue
        ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]],
                "r-", alpha=0.15, linewidth=0.6)
    for i in chosen_miss:
        o = photons["origin"][i]
        d = photons["direction"][i]
        denom = np.dot(d, det_n)
        if abs(denom) < 1e-8:
            continue
        t = np.dot(det_c - o, det_n) / denom
        if t <= 0:
            continue
        end = o + t * d
        ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]],
                color="gray", alpha=0.08, linewidth=0.4)

    # 反射段（晶体→PC）：仅画命中 PC 的光子
    n_pc = len(result.pc_hit_points)
    if n_pc > 0:
        n_show = min(max_rays, n_pc)
        idx = np.random.choice(n_pc, size=n_show, replace=False)
        for i in idx:
            a_pt = result.pc_crystal_hit_points[i]
            b_pt = result.pc_hit_points[i]
            ax.plot([a_pt[0], b_pt[0]], [a_pt[1], b_pt[1]], [a_pt[2], b_pt[2]],
                    "m-", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(f"3D Ray Paths — {cfg.crystal_type.upper()}\n"
                 f"Reflected: {result.N_reflected}  PC hit: {result.N_pc_hit}")
    ax.view_init(elev=25, azim=115)


# ---------------------------------------------------------------------------
# YZ 投影
# ---------------------------------------------------------------------------

def draw_yz(ax, result, max_rays=300, ray_mode="full"):
    """
    在 2D axes 上绘制 YZ 投影光路。

    ``ray_mode``：

    - ``"full"``（默认）：绘制全部入射光线（wafer→slit→晶体，红色）、
      未命中晶体的光线（灰色）与反射光线（晶体→PC，品红）。
    - ``"reflected"``：**只绘制「成功被反射」的那部分光线的完整路径** ——
      其入射段（wafer 原点 → 晶体命中点，红色细线）+ 反射段
      （晶体命中点 → PC 窗口，品红）。未命中、未反射的光线全部不画。
      适用于 spiral 等入射段与反射段在 YZ 投影上严重交叠、难以分辨的几何。
    """
    cfg = result.config
    det_c, det_n = result.detector_center, result.detector_normal
    cryst_c = result.crystal_center
    pc_c = result.pc_center

    ax.clear()
    # Wafer
    ax.plot([-30, 80], [0, 0], "k-", linewidth=3, label="Wafer")
    ax.plot(0, 0, "ko", markersize=8)
    # Slit
    ax.plot(det_c[1], det_c[2], "g^", markersize=12, label="Detector Slit")
    # 晶体轮廓
    cy, cz = _crystal_yz_curve(result)
    ax.plot(cy, cz, "b-", linewidth=3, label="Crystal")
    ax.plot(cryst_c[1], cryst_c[2], "bs", markersize=8)
    # PC 窗口（YZ 投影线段，沿 pc_normal 切向取半高）
    pcn_vec = np.array([result.pc_normal[1], result.pc_normal[2]])
    pcn_vec = pcn_vec / (np.linalg.norm(pcn_vec) + 1e-12)
    tang = np.array([-pcn_vec[1], pcn_vec[0]])
    hw = cfg.pc_window_size[1] / 2
    p1 = np.array([pc_c[1], pc_c[2]]) + hw * tang
    p2 = np.array([pc_c[1], pc_c[2]]) - hw * tang
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="purple", linewidth=3,
            label="PC Window")
    ax.plot(pc_c[1], pc_c[2], "ms", markersize=8)

    # ---------------- 反射光线：晶体命中点 → PC 窗口 ----------------
    n_pc = len(result.pc_hit_points)
    idx = (np.random.choice(n_pc, size=min(max_rays, n_pc), replace=False)
           if n_pc > 0 else np.array([], dtype=int))

    if ray_mode == "reflected" and len(idx):
        # 只画「成功被反射」的光线：入射段 + 反射段拼成完整路径
        origins_after_slit = result.photons_slit["origin"]
        pc2slit = np.asarray(result.pc_hit_slit_indices, dtype=int)
        for k, i in enumerate(idx):
            slit_i = int(pc2slit[i])
            o = origins_after_slit[slit_i]
            a_pt = result.pc_crystal_hit_points[i]
            b_pt = result.pc_hit_points[i]
            ax.plot([o[1], a_pt[1]], [o[2], a_pt[2]], "r-",
                    alpha=0.45, linewidth=0.7,
                    label="Incident (reflected)" if k == 0 else None)
            ax.plot([a_pt[1], b_pt[1]], [a_pt[2], b_pt[2]],
                    "m-", alpha=0.75, linewidth=0.9,
                    label="Reflected" if k == 0 else None)
    else:
        # ---------------- 入射段：wafer → slit → 晶体 ----------------
        photons = result.photons_slit
        N_after_slit = len(photons)
        hit_pts = result.hit_points_crystal
        hit_mask_slit = result.hit_mask_slit

        # 命中晶体映射
        hit_slit_idx = np.where(hit_mask_slit)[0]
        hit_pt_for_slit = {int(hit_slit_idx[i]): hit_pts[i]
                           for i in range(len(hit_slit_idx))} if len(hit_pts) else {}

        n_hit = int(hit_mask_slit.sum())
        n_show_hit = min(max_rays, n_hit)
        n_show_miss = min(max(0, max_rays - n_show_hit), N_after_slit - n_hit)
        chosen_hit = (np.random.choice(hit_slit_idx, size=n_show_hit, replace=False)
                      if n_show_hit else np.array([], dtype=int))
        miss_idx = np.setdiff1d(np.arange(N_after_slit), hit_slit_idx)
        chosen_miss = (np.random.choice(miss_idx, size=n_show_miss, replace=False)
                       if n_show_miss and len(miss_idx) else np.array([], dtype=int))

        for i in chosen_hit:
            o = photons["origin"][i]
            end = hit_pt_for_slit.get(int(i))
            if end is None:
                continue
            ax.plot([o[1], end[1]], [o[2], end[2]], "r-",
                    alpha=0.12, linewidth=0.6)
        for i in chosen_miss:
            o = photons["origin"][i]
            d = photons["direction"][i]
            denom = np.dot(d, det_n)
            if abs(denom) < 1e-8:
                continue
            t = np.dot(det_c - o, det_n) / denom
            if t <= 0:
                continue
            end = o + t * d
            ax.plot([o[1], end[1]], [o[2], end[2]], color="gray",
                    alpha=0.06, linewidth=0.4)

        # ---------------- 反射段：晶体命中点 → PC 窗口 ----------------
        for k, i in enumerate(idx):
            a_pt = result.pc_crystal_hit_points[i]
            b_pt = result.pc_hit_points[i]
            ax.plot([a_pt[1], b_pt[1]], [a_pt[2], b_pt[2]],
                    "m-", alpha=0.5, linewidth=0.8,
                    label="Reflected" if k == 0 else None)

    ax.set_xlabel("Y [mm]")
    ax.set_ylabel("Z [mm]")
    title = f"YZ Projection — {cfg.crystal_type.upper()}\n" \
            f"PC hit: {result.N_pc_hit} / {result.N_emitted:,}"
    if ray_mode == "reflected":
        title += "\n(reflected paths only)"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


# ---------------------------------------------------------------------------
# 概览统计图
# ---------------------------------------------------------------------------

def draw_overview(fig, result):
    """在 fig（1x3 子图）上绘制统计概览。"""
    cfg = result.config
    axes = fig.subplots(1, 3)

    # 1a 漏斗图
    ax = axes[0]
    stages = ["Emitted", "After Slit", "Hit Crystal", "Reflected", "PC Hit"]
    counts = [result.N_emitted, result.N_after_slit, result.N_hit_crystal,
              result.N_reflected, result.N_pc_hit]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    bars = ax.bar(stages, counts, color=colors, edgecolor="white", linewidth=1.2)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Photon Count")
    ax.set_title("Photon Pipeline")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.tick_params(axis="x", rotation=30)

    # 1b PC 窗口面分布
    ax = axes[1]
    if result.N_pc_hit > 0:
        pts = result.pc_hit_points
        u, v = _rect_basis(result.pc_normal)
        rel = pts - result.pc_center
        lu = np.dot(rel, u)
        lv = np.dot(rel, v)
        ax.scatter(lu, lv, s=1, alpha=0.3, c="purple")
        w, h = cfg.pc_window_size
        ax.add_patch(plt.Rectangle((-w / 2, -h / 2), w, h, fill=False,
                                   edgecolor="red", linewidth=2))
    else:
        ax.text(0.5, 0.5, "No PC hit", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
    ax.set_xlabel("u (width) [mm]")
    ax.set_ylabel("v (height) [mm]")
    ax.set_title(f"Photon Distribution on PC Window\n(N={result.N_pc_hit})")
    ax.set_aspect("equal")

    # 1c 反射光子能量分布
    ax = axes[2]
    if result.N_reflected > 0:
        ax.hist(result.photons_reflected["energy"], bins=50, color="#F44336",
                alpha=0.7, edgecolor="white")
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Count")
        ax.set_title(f"Reflected Photon Energy\n(N={result.N_reflected})")
    else:
        ax.text(0.5, 0.5, "No reflected\nphotons", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title("Reflected Photon Energy")

    fig.suptitle(f"XRF: {cfg.crystal_type.upper()} Crystal — "
                 f"θ_B={result.bragg_angle_deg:.2f}°, E={cfg.energy_eV} eV",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()


# ---------------------------------------------------------------------------
# 一键生成三张图
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PC 命中光子的 wafer 原点追溯
# ---------------------------------------------------------------------------

def _draw_region_outline(ax, region, color, label, linestyle="-"):
    """在 XY 平面上绘制一个区域（矩形 / 圆形）的轮廓。"""
    rtype = str(region.get("type", "rect")).lower()
    if rtype == "rect":
        x0, x1 = sorted(region["x_range"])
        y0, y1 = sorted(region["y_range"])
        patch = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                              edgecolor=color, linewidth=2.0,
                              linestyle=linestyle, label=label)
    elif rtype == "circle":
        cx, cy = region["center"]
        patch = plt.Circle((cx, cy), region["radius"], fill=False,
                           edgecolor=color, linewidth=2.0,
                           linestyle=linestyle, label=label)
    else:
        return
    ax.add_patch(patch)


def draw_wafer_origins(fig, result, max_points=20000, regions=None):
    """
    在 wafer 面 (XY) 上绘制 PC 命中光子的发射原点分布。

    - 若配置使用 spot_profile，绘制强度热图作为底图，原点散点叠加其上。
    - 否则仅绘制原点散点（底图显示 spot_radius 圆形区域）。
    - ``regions`` 非空时，在图上绘制每个区域轮廓，并附加一张柱状图展示
      各区域内「PC 接收光强占全部 PC 接收光强的比例」（``pc_fraction``）。
    """
    from .simulate import evaluate_regions

    cfg = result.config
    stats = summary = None
    if regions:
        stats, summary = evaluate_regions(result, regions)
    if regions:
        ax, ax_bar = fig.subplots(
            1, 2, gridspec_kw={"width_ratios": [1.2, 1.0]}
        )
    else:
        ax = fig.add_subplot(111)
        ax_bar = None

    profile = cfg.spot_profile
    origins = result.wafer_origins_pc
    n_pc = len(origins)

    # 底图：强度分布热图
    if profile is not None:
        prof, xr, yr = profile
        im = ax.imshow(
            prof, extent=[xr[0], xr[1], yr[0], yr[1]],
            origin="lower", aspect="auto", cmap="viridis", alpha=0.55,
        )
        fig.colorbar(im, ax=ax, label="Profile intensity")
        xmin, xmax = xr
        ymin, ymax = yr
    else:
        r = cfg.spot_radius
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                "k--", linewidth=1.2, label="Spot radius")
        xmin, xmax = -1.2 * r, 1.2 * r
        ymin, ymax = -1.2 * r, 1.2 * r

    # PC 命中光子的 wafer 原点
    if n_pc > 0:
        pts = origins
        if n_pc > max_points:
            sel = np.random.choice(n_pc, size=max_points, replace=False)
            pts = origins[sel]
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c="red",
                   alpha=0.45, linewidths=0,
                   label=f"PC-hit wafer origins (N={n_pc:,})")

    # 区域轮廓
    if regions:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, region in enumerate(regions):
            _draw_region_outline(
                ax, region,
                color=colors[i % 10],
                label=stats[i]["label"] if stats else f"region{i + 1}",
                linestyle="--",
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax.set_title(
        f"PC Photon Wafer Origins — {cfg.crystal_type.upper()}\n"
        f"PC hit: {result.N_pc_hit:,} / {result.N_emitted:,}"
    )

    # 各区域的 PC 接收光强占比柱状图
    if ax_bar is not None:
        labels = [s["label"] for s in stats]
        vals = [(s["pc_fraction"] or 0.0) * 100 for s in stats]
        bars = ax_bar.barh(range(len(labels)), vals,
                           color=[plt.cm.tab10(i % 10) for i in range(len(labels))],
                           alpha=0.8)
        ax_bar.set_yticks(range(len(labels)))
        ax_bar.set_yticklabels(labels, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Share of total PC-received intensity [%]")
        sub = (f"total PC hits = {summary['n_pc_total']:,}"
               f"  |  N_emitted = {summary['n_emit_total']:,}")
        ax_bar.set_title(
            "Per-region share of PC-received intensity\n" + sub,
            fontsize=10,
        )
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                        f" {v:.3f}%", va="center", fontsize=8)
        ax_bar.grid(True, axis="x", alpha=0.3)
        xmax_bar = max(vals) if vals and max(vals) > 0 else 1.0
        ax_bar.set_xlim(0, xmax_bar * 1.35)

    fig.tight_layout()
    return ax


def build_figures(result, max_vis_rays=300, regions=None, yz_ray_mode=None):
    """
    生成 overview / 3D / YZ / wafer-origins 四张 Figure，返回 dict。

    ``yz_ray_mode``：YZ 投影图的光线显示模式。

    - ``"full"``：全部入射光线 + 未命中光线 + 反射光线
    - ``"reflected"``：只显示「成功被反射」的光线的完整路径（入射段 + 反射段）
    - ``None``（默认）：自动判断 —— ``spiral`` 晶体因入射段与反射段在 YZ 投影上
      严重交叠，默认用 ``"reflected"``；其余晶体用 ``"full"``
    """
    if yz_ray_mode is None:
        yz_ray_mode = ("reflected" if result.config.crystal_type == "spiral"
                       else "full")

    fig1 = plt.figure(figsize=(18, 5))
    draw_overview(fig1, result)
    fig2 = plt.figure(figsize=(12, 9))
    ax3d = fig2.add_subplot(111, projection="3d")
    draw_3d(ax3d, result, max_rays=max_vis_rays)
    fig2.tight_layout()
    fig3 = plt.figure(figsize=(12, 8))
    ax_yz = fig3.add_subplot(111)
    draw_yz(ax_yz, result, max_rays=max_vis_rays, ray_mode=yz_ray_mode)
    fig3.tight_layout()
    fig4 = plt.figure(figsize=(16, 7) if regions else (9, 8))
    draw_wafer_origins(fig4, result, regions=regions)
    return {"overview": fig1, "3d_paths": fig2, "yz_projection": fig3,
            "wafer_origins": fig4}
