#!/usr/bin/env python3
"""
debug_tool.py
=============
XRF 荧光模拟统一调试接口（wafer → detector slit → 晶体 → 正比计数器窗口）。

支持三种晶体类型的一键模拟与可视化：
    1. flat          — 平晶
    2. cylindrical   — 柱面 Johann 型弯晶
    3. spiral        — 对数螺线弯晶

使用方式
--------
1. 作为函数调用（推荐，在 Jupyter / 脚本中使用）::

        from xrf_sim.debug_tool import run_debug

        result = run_debug(crystal_type="flat", energy_eV=183, ...)

2. 命令行调用::

        python -m xrf_sim.debug_tool --crystal flat --energy 183 ...
""" 

import os
import sys
import numpy as np

# ------------------------------------------------------------------
# 在导入 pyplot 之前，确保选用交互式后端。
# 优先级：Qt5Agg > TkAgg > WebAgg；无可用交互后端时回退到 Agg。
# ------------------------------------------------------------------
import matplotlib


def _probe_backend(backend_name: str) -> bool:
    """在独立子进程中测试指定后端是否可用。"""
    import subprocess
    code = (
        f"import matplotlib; matplotlib.use('{backend_name}'); "
        "import matplotlib.pyplot as p; p.figure(); p.close('all'); print('OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
        return result.returncode == 0 and b'OK' in result.stdout
    except Exception:
        return False


_JUPYTER_BACKENDS = frozenset([
    'module://matplotlib_inline.backend_inline',
    'module://ipympl.backend_nbagg',
    'nbagg', 'notebook', 'inline',
])


def _setup_interactive_backend():
    """按优先级选择交互式后端，返回后端名称。"""
    current = matplotlib.get_backend().lower()
    if current in _JUPYTER_BACKENDS or current.startswith('module://'):
        return current
    for backend in ['Qt5Agg', 'TkAgg', 'WebAgg']:
        if _probe_backend(backend):
            matplotlib.use(backend)
            return backend
    matplotlib.use('Agg')
    print(
        '[debug_tool] WARNING: No interactive backend available.\n'
        '  Install PyQt5 to enable interactive display:\n'
        '    pip install PyQt5\n'
        '  Or use --save_dir <path> to save figures to files.',
        file=sys.stderr
    )
    return 'agg'


_ACTIVE_BACKEND = _setup_interactive_backend()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .simulate import simulate, SimulationConfig, save_pc_photons, evaluate_regions
from .plotting import build_figures


# =====================================================================
# 核心调试入口
# =====================================================================

def _write_params_md(path: str, cfg, result, regions=None) -> None:
    """将本次模拟使用的参数、结果与区域统计写入 Markdown 文件。"""
    from datetime import datetime
    lines = []
    lines.append("# XRF 模拟参数记录")
    lines.append("")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 晶体类型：`{cfg.crystal_type}`")
    lines.append("")
    lines.append("## 输入参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 能量 energy_eV | {cfg.energy_eV} eV |")
    lines.append(f"| 晶面间距 d_spacing | {cfg.d_spacing} m |")
    lines.append(f"| 探测器距离 detector_distance | {cfg.detector_distance} mm |")
    lines.append(f"| 探测器掠射角 detector_theta_deg | {cfg.detector_theta_deg}° |")
    lines.append(f"| 探测器尺寸 detector_size | {cfg.detector_size} mm |")
    lines.append(f"| 晶体到探测器距离 crystal_distance | {cfg.crystal_distance} mm |")
    lines.append(f"| 晶体尺寸 crystal_size | {cfg.crystal_size} mm |")
    lines.append(f"| 光子总数 N_photons | {cfg.N_photons:,} |")
    lines.append(f"| 反射率 FWHM fwhm_deg | {cfg.fwhm_deg}° |")
    lines.append(f"| 光斑半径 spot_radius | {cfg.spot_radius} mm |")
    lines.append(f"| Soller 狭缝 is_flatted | {cfg.is_flatted} |")
    lines.append(f"| 曲率半径 ROC | {cfg.ROC} mm |")
    lines.append(f"| slit 到晶体距离 a_crystal | {cfg.a_crystal} mm |")
    lines.append(f"| 锥形发射半角 cone_angle_deg | {cfg.cone_angle_deg}° |")
    lines.append(f"| PC 窗口尺寸 pc_window_size | {cfg.pc_window_size} mm |")
    lines.append(f"| 针孔中心 pinhole_center | {cfg.pinhole_center} |")
    lines.append(f"| 针孔半径 pinhole_radius | {cfg.pinhole_radius} mm |")
    lines.append(f"| 强度分布 spot_profile | {'是' if cfg.spot_profile is not None else '否'} |")
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append(f"- 布拉格角：{result.bragg_angle_deg:.4f}°")
    lines.append(f"- 发射光子数 N_emitted：{result.N_emitted:,}")
    lines.append(f"- 通过狭缝 N_after_slit：{result.N_after_slit:,}")
    lines.append(f"- 命中晶体 N_hit_crystal：{result.N_hit_crystal:,}")
    lines.append(f"- 反射光子 N_reflected：{result.N_reflected:,}")
    lines.append(f"- PC 命中 N_pc_hit：{result.N_pc_hit:,}")
    lines.append(f"- 总效率：{result.efficiency:.6%}")
    lines.append(f"- PC 效率：{result.pc_efficiency:.6%}")
    # 指定区域的 PC 接收光强占比
    if regions:
        try:
            stats, summary = evaluate_regions(result, regions)

            def _p(v):
                return f"{v:.4%}" if v is not None else "—"

            lines.append("")
            lines.append("## 各区域 PC 接收光强占全部 PC 光强的比例")
            lines.append("")
            lines.append("指标说明：")
            lines.append("")
            lines.append("- **pc_fraction**（核心）：该区域的 PC 接收光强占「全部 PC 接收光强」的比例")
            lines.append("  = N_pc(区域) / N_pc_total。各区域互斥且铺满发光区时，全部区域之和 = 100%")
            lines.append("- pc_intensity_fraction：该区域 PC 光强占「全部发射光强」的比例")
            lines.append("  = N_pc(区域) / N_emit_total（互斥铺满时之和 ≈ 总 PC 效率）")
            lines.append("- intensity_fraction：该区域占 profile 总强度的比例 = I(区域) / I_total")
            lines.append("- collection_efficiency：该区域的收集效率 = N_pc(区域) / N_emit(区域)")
            lines.append("")
            lines.append("| 区域 | **占全部PC光强** | 占全部发射光强 | 占profile强度 | 区域收集效率 | N_pc(区域) | N_emit(区域) |")
            lines.append("|------|------|------|------|------|------|------|")
            for s in stats:
                lines.append(
                    f"| {s['label']} | **{_p(s['pc_fraction'])}** | "
                    f"{_p(s['pc_intensity_fraction'])} | {_p(s['intensity_fraction'])} | "
                    f"{_p(s['collection_efficiency'])} | "
                    f"{s['n_pc_in_region']:,} | {s['n_emit_in_region']:,} |"
                )
            total_region = sum(s["n_pc_in_region"] for s in stats)
            lines.append("")
            lines.append(f"- 各区域 PC 命中合计：{total_region:,}"
                         f"（总 PC 命中 {summary['n_pc_total']:,}）"
                         f"→ 合计占总 PC 光强 {_p(summary['pc_fraction_sum'])}")
            lines.append(f"- 总 PC 效率：{_p(summary['pc_efficiency'])}"
                         f"（总发射光子 {summary['n_emit_total']:,}）")
            if (summary["pc_fraction_sum"] is not None
                    and abs(summary["pc_fraction_sum"] - 1.0) > 1e-9):
                lines.append("")
                lines.append("> 各区域之和不为 100%：区域可能相互重叠，"
                             "或未铺满整个发光区。")
        except Exception as e:
            lines.append(f"\n> 区域统计失败：{e}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_debug(
    crystal_type: str = "flat",
    energy_eV: float = 183.0,
    d_spacing: float = 8.95e-9,
    detector_distance: float = 90,
    detector_theta_deg: float = 30.0,
    detector_size: tuple = (22, 19),
    crystal_distance: float = 30.0,
    crystal_size: tuple = (30, 40),
    N_photons: int = 10_000_000,
    fwhm_deg: float = 0.8,
    spot_radius: float = 20.0,
    is_flatted: bool = False,
    ROC: float = None,
    a_crystal: float = None,
    spot_profile: tuple = None,
    max_vis_rays: int = 300,
    save_dir: str = None,
    pinhole_center=(0.0, 0.0, 14.02),
    pinhole_radius: float = 11.45,
    pc_window_size: tuple = (30, 25),
    cone_angle_deg: float = None,
    profile_file: str = None,
    regions: list = None,
    yz_ray_mode: str = None,
) -> dict:
    """
    执行一次完整的 XRF 荧光模拟（含 PC 窗口）并生成可视化。

    返回 dict 包含各级光子数、效率、PC 命中数、SimulationResult 与 figures。

    ``yz_ray_mode``：YZ 投影图光线显示模式，``"full"`` / ``"reflected"``。
    ``None`` 时自动判断（spiral 默认 ``"reflected"``，其余默认 ``"full"``）。
    """
    cfg = SimulationConfig(
        crystal_type=crystal_type,
        energy_eV=energy_eV,
        d_spacing=d_spacing,
        detector_distance=detector_distance,
        detector_theta_deg=detector_theta_deg,
        detector_size=tuple(detector_size),
        crystal_distance=crystal_distance,
        crystal_size=tuple(crystal_size),
        N_photons=N_photons,
        fwhm_deg=fwhm_deg,
        spot_radius=spot_radius,
        is_flatted=is_flatted,
        ROC=ROC,
        a_crystal=a_crystal,
        spot_profile=spot_profile,
        pinhole_center=pinhole_center,
        pinhole_radius=pinhole_radius,
        pc_window_size=tuple(pc_window_size),
        cone_angle_deg=cone_angle_deg,
    )

    print("=" * 70)
    print(f"  XRF Debug Tool — crystal_type = {crystal_type}")
    print("=" * 70)

    result = simulate(cfg)

    print("-" * 70)
    print(f"  [Result] N_emitted      = {result.N_emitted:,}")
    print(f"  [Result] N_after_slit   = {result.N_after_slit:,}")
    print(f"  [Result] N_hit_crystal  = {result.N_hit_crystal:,}")
    print(f"  [Result] N_reflected    = {result.N_reflected:,}")
    print(f"  [Result] N_pc_hit       = {result.N_pc_hit:,}")
    print(f"  [Result] Efficiency     = {result.efficiency:.6%}")
    print(f"  [Result] PC efficiency  = {result.pc_efficiency:.6%}")
    # 指定区域的 PC 接收光强占比
    if regions:
        try:
            stats, summary = evaluate_regions(result, regions)
            print("  [Regions] 各区域 PC 接收光强占全部 PC 光强的比例：")
            for s in stats:
                frac = (f"{s['pc_fraction']:.4%}"
                        if s["pc_fraction"] is not None else "—")
                eta = (f"{s['collection_efficiency']:.4%}"
                       if s["collection_efficiency"] is not None else "—")
                print(f"    - {s['label']}: {frac}  "
                      f"(区域收集效率 {eta}, N_pc={s['n_pc_in_region']:,})")
            if (summary["pc_fraction_sum"] is not None
                    and abs(summary["pc_fraction_sum"] - 1.0) > 1e-9):
                print(f"  [Regions] 各区域合计占总 PC 光强 "
                      f"{summary['pc_fraction_sum']:.4%}（未铺满整个发光区或存在重叠）")
        except Exception as e:
            print(f"  [WARN] 区域统计失败: {e}")
    print("=" * 70)

    figures = build_figures(result, max_vis_rays=max_vis_rays, regions=regions,
                            yz_ray_mode=yz_ray_mode)

    if save_dir is not None:
        # 每次运行结果放入以日期命名的子文件夹，避免覆盖
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir, stamp)
        os.makedirs(out_dir, exist_ok=True)

        for name, fig in figures.items():
            path = os.path.join(out_dir, f"debug_{crystal_type}_{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  [Saved] {path}")
        # 保存 PC 窗口接收到的光子数据（uv 坐标系）
        pc_csv = os.path.join(out_dir, f"debug_{crystal_type}_pc_photons.csv")
        save_pc_photons(result, pc_csv)
        print(f"  [Saved] {pc_csv}")

        # 记录本次模拟使用的参数到 .md 文件
        md_path = os.path.join(out_dir, "params.md")
        _write_params_md(md_path, cfg, result, regions=regions)
        print(f"  [Saved] {md_path}")

        # 复制所使用的光斑文件 + 生成光斑可视化图
        if profile_file is not None and os.path.exists(profile_file):
            import shutil
            base = os.path.basename(profile_file)
            dst = os.path.join(out_dir, base)
            shutil.copy2(profile_file, dst)
            print(f"  [Saved] {dst}")
            try:
                from .view_profile import view_profile
                view_profile(
                    filepath=profile_file,
                    x_range=cfg.spot_profile[1] if cfg.spot_profile else None,
                    y_range=cfg.spot_profile[2] if cfg.spot_profile else None,
                    save_dir=out_dir,
                )
            except Exception as e:
                print(f"  [WARN] 生成 profile 可视化失败: {e}")
    else:
        if _ACTIVE_BACKEND == 'agg':
            print(
                '[debug_tool] WARNING: Running in non-interactive (Agg) backend.\n'
                '  No window will be shown. Please install PyQt5 or tkinter,\n'
                '  or use --save_dir <path> to save figures to files.',
                file=sys.stderr
            )
        else:
            plt.ion()
            plt.show()
            plt.pause(0.001)
            input('  [Press Enter to close all figures and exit] ')
            plt.close('all')

    return {
        "crystal_type":        crystal_type,
        "N_emitted":           result.N_emitted,
        "N_after_slit":        result.N_after_slit,
        "N_hit_crystal":       result.N_hit_crystal,
        "N_reflected":         result.N_reflected,
        "N_pc_hit":            result.N_pc_hit,
        "efficiency":          result.efficiency,
        "pc_efficiency":       result.pc_efficiency,
        "bragg_angle_deg":     result.bragg_angle_deg,
        "result":              result,
        "figures":             figures,
    }


# =====================================================================
# 命令行入口
# =====================================================================

def _parse_args():
    """解析命令行参数。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="XRF 荧光模拟统一调试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--crystal", type=str, default="flat",
                        choices=["flat", "cylindrical", "spiral"],
                        help="晶体类型 (default: flat)")
    parser.add_argument("--energy", type=float, default=183.0,
                        help="光子能量 eV (default: 183)")
    parser.add_argument("--d_spacing", type=float, default=8.95e-9,
                        help="晶面间距 m (default: 8.95e-9)")
    parser.add_argument("--det_dist", type=float, default=124.0,
                        help="Wafer 到 detector slit 距离 mm (default: 124)")
    parser.add_argument("--det_deg", type=float, default=30.0,
                        help="探测器掠射角 deg (default: 30)")
    parser.add_argument("--det_size", type=float, nargs=2, default=[22, 19],
                        help="探测器尺寸 width height mm (default: 22 19)")
    parser.add_argument("--cryst_dist", type=float, default=30.0,
                        help="晶体到探测器中心距离 mm (default: 30)")
    parser.add_argument("--cryst_size", type=float, nargs=2, default=[30, 40],
                        help="晶体尺寸 width height/arc mm (default: 30 40)")
    parser.add_argument("--N", type=int, default=None,
                        help="模拟光子总数 (default: 由 run_debug 决定)")
    parser.add_argument("--fwhm", type=float, default=0.8,
                        help="反射率 FWHM deg (default: default: 0.8)")
    parser.add_argument("--spot_radius", type=float, default=None,
                        help="光斑半径 mm (default: 由 run_debug 决定)")
    parser.add_argument("--ROC", type=float, default=None,
                        help="弯晶曲率半径 mm (cylindrical only)")
    parser.add_argument("--a_crystal", type=float, default=None,
                        help="Slit 到晶体中心距离 mm (spiral only)")
    parser.add_argument("--flatted", action="store_true",
                        help="启用 Soller 狭缝")
    parser.add_argument("--pc_size", type=float, nargs=2, default=[30, 25],
                        help="PC 窗口尺寸 width height mm (default: 25 5)")
    parser.add_argument("--max_rays", type=int, default=500,
                        help="可视化最多光线数 (default: 300)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="图片保存目录；不指定则直接弹出交互式窗口")
    parser.add_argument("--profile", type=str, default=None,
                        help="强度分布文件路径 (.npy/.csv/.txt)；提供后按该分布采样光斑")
    parser.add_argument("--profile_x_range", type=float, nargs=2, default=None,
                        help="强度分布 x 范围 xmin xmax mm (default: -25 25)")
    parser.add_argument("--profile_y_range", type=float, nargs=2, default=None,
                        help="强度分布 y 范围 ymin ymax mm (default: -25 25)")
    parser.add_argument("--cone", type=float, default=None,
                        help="发射方向锥形半角 deg；提供后以探测器方向为中心采样，"
                             "大幅提高命中率 (default: 上半球均匀采样)")
    parser.add_argument("--region_rect", type=float, nargs=4, action="append",
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="统计区域（矩形）：xmin xmax ymin ymax mm；"
                             "可重复指定多个区域")
    parser.add_argument("--region_circle", type=float, nargs=3, action="append",
                        metavar=("CX", "CY", "R"),
                        help="统计区域（圆形）：cx cy r mm；"
                             "可重复指定多个区域")
    yz_group = parser.add_mutually_exclusive_group()
    yz_group.add_argument("--yz_full", dest="yz_ray_mode", action="store_const",
                          const="full", default=None,
                          help="YZ 投影图显示完整光路（含全部入射光线）；"
                               "默认 spiral 只显示被反射光线的完整路径")
    yz_group.add_argument("--yz_reflected", dest="yz_ray_mode",
                          action="store_const", const="reflected",
                          help="YZ 投影图只显示「成功被反射」的光线："
                               "入射段 + 反射段")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # 若提供强度分布文件，则加载
    spot_profile = None
    if args.profile is not None:
        from .core.spot_profile import load_profile_file
        x_range = tuple(args.profile_x_range) if args.profile_x_range else None
        y_range = tuple(args.profile_y_range) if args.profile_y_range else None
        spot_profile = load_profile_file(args.profile, x_range, y_range)
        print(f"[CLI] 已加载强度分布: {args.profile} shape={spot_profile[0].shape}")

    # 组装区域列表（矩形 / 圆形，可多选）
    regions = []
    for r in (args.region_rect or []):
        x0, x1, y0, y1 = r
        regions.append({"type": "rect", "x_range": (x0, x1),
                        "y_range": (y0, y1)})
    for c in (args.region_circle or []):
        cx, cy, rad = c
        regions.append({"type": "circle", "center": (cx, cy),
                        "radius": rad})
    if regions:
        print(f"[CLI] 已指定 {len(regions)} 个统计区域")

    # 构建 kwargs：CLI 中为 None 的参数不传入，回退到 run_debug 默认值
    kwargs = dict(
        crystal_type=args.crystal,
        energy_eV=args.energy,
        d_spacing=args.d_spacing,
        detector_distance=args.det_dist,
        detector_theta_deg=args.det_deg,
        detector_size=tuple(args.det_size),
        crystal_distance=args.cryst_dist,
        crystal_size=tuple(args.cryst_size),
        fwhm_deg=args.fwhm,
        is_flatted=args.flatted,
        spot_profile=spot_profile,
        max_vis_rays=args.max_rays,
        save_dir=args.save_dir,
        pc_window_size=tuple(args.pc_size),
        profile_file=args.profile,
        regions=regions or None,
    )
    # 仅当 CLI 显式提供时才覆盖（否则用 run_debug 默认值）
    if args.N is not None:
        kwargs["N_photons"] = args.N
    if args.spot_radius is not None:
        kwargs["spot_radius"] = args.spot_radius
    if args.ROC is not None:
        kwargs["ROC"] = args.ROC
    if args.a_crystal is not None:
        kwargs["a_crystal"] = args.a_crystal
    if args.cone is not None:
        kwargs["cone_angle_deg"] = args.cone
    if args.yz_ray_mode is not None:
        kwargs["yz_ray_mode"] = args.yz_ray_mode

    run_debug(**kwargs)
