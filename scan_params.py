#!/usr/bin/env python3
"""
xrf_sim/experiments/scripts/scan_params.py
==========================================
通用参数扫描脚本（基于 ``simulate()``，**含 PC 窗口**）。

与 ``scan_flat_angle`` / ``scan_curved_roc`` / ``scan_curved_distance`` 的区别：
那三个脚本直接调用 ``BraggDiffraction``，只统计到「反射光子数」；本脚本使用
完整管线 ``simulate()``，因此额外输出 slit / 晶体命中 / 反射 / **PC 命中**
各级计数与效率，并可直接衔接 ``evaluate_regions`` 的区域占比统计。

用法
----
1) 单参数、等间距取值（``start:stop:num``）::

    python -m xrf_sim.experiments.scripts.scan_params --crystal spiral --a_crystal 160 ^
        --sweep "fwhm_deg=0.01:0.2:10" --N 200000

2) 单参数、显式取值列表（``v1,v2,v3``）::

    python -m xrf_sim.experiments.scripts.scan_params --crystal flat ^
        --sweep "detector_theta_deg=25,30,35,40" --N 200000

3) 多参数网格（``--sweep`` 可重复，按笛卡尔积展开）::

    python -m xrf_sim.experiments.scripts.scan_params --crystal cylindrical ^
        --sweep "ROC=100,150,200" --sweep "fwhm_deg=0.1,0.5" --N 100000

键名必须是 ``SimulationConfig`` 的字段名（可通过 ``--list-params`` 查看）。
``detector_size`` / ``crystal_size`` / ``pc_window_size`` 是元组，不适合直接扫描。

输出
----
- ``<out_dir>/scan_<crystal>_<params>.csv``：每次模拟一行的明细（追加写入）
- ``<out_dir>/scan_<crystal>_<params>.png``：一维为折线图，二维为热图
"""

import os
import sys
import itertools
import argparse
import dataclasses

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xrf_sim.simulate import simulate, SimulationConfig, evaluate_regions
from xrf_sim.experiments.scanner import run_scan

DEFAULT_OUT_DIR = os.path.join(_ROOT, "sim_output", "scans")


# =====================================================================
# 参数解析辅助
# =====================================================================

def _coerce(text: str):
    """把字符串转成 int / float（失败则保留原字符串）。"""
    text = text.strip()
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def parse_sweep(spec: str):
    """
    解析一条 ``--sweep`` 规则，返回 ``(name, values)``。

    支持两种写法：
    - ``name=v1,v2,v3``      显式取值列表
    - ``name=start:stop:num`` 等间距取值（含端点）
    """
    if "=" not in spec:
        raise ValueError(f"--sweep 格式错误（缺少 '='）：{spec!r}；"
                         f"应为 name=v1,v2,v3 或 name=start:stop:num")
    name, _, rhs = spec.partition("=")
    name = name.strip()
    rhs = rhs.strip()
    if not name or not rhs:
        raise ValueError(f"--sweep 格式错误：{spec!r}")

    if ":" in rhs:
        parts = [p.strip() for p in rhs.split(":")]
        if len(parts) != 3:
            raise ValueError(f"--sweep 区间写法应为 start:stop:num，收到 {rhs!r}")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        if num < 1:
            raise ValueError(f"--sweep 区间取值个数必须 >= 1，收到 {num}")
        values = np.linspace(start, stop, num).tolist()
        # 整型参数保持整型
        if all(float(v).is_integer() for v in values) and "." not in parts[0]:
            values = [int(round(v)) for v in values]
    else:
        values = [_coerce(v) for v in rhs.split(",") if v.strip()]

    if not values:
        raise ValueError(f"--sweep 未解析出任何取值：{spec!r}")
    return name, values


def _validate_param_names(names):
    """校验扫描参数名是否都是 SimulationConfig 的字段。"""
    valid = {f.name for f in dataclasses.fields(SimulationConfig)}
    bad = [n for n in names if n not in valid]
    if bad:
        raise SystemExit(
            f"[ERROR] 以下参数不是 SimulationConfig 的字段：{bad}\n"
            f"        可用字段：{', '.join(sorted(valid))}"
        )


# =====================================================================
# 单次模拟
# =====================================================================

def make_runner(args, spot_profile, regions, total=None):
    """返回 ``single_run(overrides: dict) -> dict``。"""
    counter = {"i": 0}

    base = dict(
        crystal_type=args.crystal,
        energy_eV=args.energy,
        d_spacing=args.d_spacing,
        detector_distance=args.det_dist,
        detector_theta_deg=args.det_deg,
        detector_size=tuple(args.det_size),
        crystal_distance=args.cryst_dist,
        crystal_size=tuple(args.cryst_size),
        N_photons=args.N,
        fwhm_deg=args.fwhm,
        spot_radius=args.spot_radius,
        is_flatted=args.flatted,
        ROC=args.ROC,
        a_crystal=args.a_crystal,
        spot_profile=spot_profile,
        pc_window_size=tuple(args.pc_size),
        cone_angle_deg=args.cone,
        seed=args.seed,
        verbose=False,
    )

    def single_run(overrides: dict) -> dict:
        counter["i"] += 1
        desc = ", ".join(f"{k}={v}" for k, v in overrides.items())
        head = f"[{counter['i']}/{total}]" if total else f"[{counter['i']}]"
        print(f"{head} {desc}", flush=True)

        cfg = SimulationConfig(**{**base, **overrides})
        result = simulate(cfg)

        row = {
            "crystal_type": cfg.crystal_type,
            "energy_eV": cfg.energy_eV,
            "d_spacing": cfg.d_spacing,
            "detector_distance": cfg.detector_distance,
            "detector_theta_deg": cfg.detector_theta_deg,
            "crystal_distance": cfg.crystal_distance,
            "crystal_size_w": cfg.crystal_size[0],
            "crystal_size_h": cfg.crystal_size[1],
            "pc_window_w": cfg.pc_window_size[0],
            "pc_window_h": cfg.pc_window_size[1],
            "N_photons": cfg.N_photons,
            "fwhm_deg": cfg.fwhm_deg,
            "spot_radius": cfg.spot_radius,
            "is_flatted": cfg.is_flatted,
            "ROC": cfg.ROC,
            "a_crystal": cfg.a_crystal,
            "bragg_angle_deg": result.bragg_angle_deg,
            "N_after_slit": result.N_after_slit,
            "N_hit_crystal": result.N_hit_crystal,
            "N_reflected": result.N_reflected,
            "N_pc_hit": result.N_pc_hit,
            "efficiency": result.efficiency,
            "pc_efficiency": result.pc_efficiency,
        }
        # 扫描参数原样记录（避免被上面的同名列覆盖成基础值）
        for k, v in overrides.items():
            row[k] = v

        # 可选：区域 PC 光强占比
        if regions:
            stats, _ = evaluate_regions(result, regions)
            for s in stats:
                row[f"pc_frac[{s['label']}]"] = s["pc_fraction"]
                row[f"n_pc[{s['label']}]"] = s["n_pc_in_region"]
        return row

    return single_run


# =====================================================================
# 绘图
# =====================================================================

def plot_scan(df, names, out_path):
    """一维 → 折线图；二维 → 热图；更高维不画图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("N_hit_crystal", "Crystal hits"),
        ("N_reflected", "Reflected photons"),
        ("N_pc_hit", "PC hits"),
        ("pc_efficiency", "PC efficiency [%]"),
    ]

    if len(names) == 1:
        key = names[0]
        xs = np.asarray(df[key], dtype=float)
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
        for ax, (col, label) in zip(axes, metrics):
            ys = np.asarray(df[col], dtype=float)
            if col == "pc_efficiency":
                ys = ys * 100.0
            ax.plot(xs, ys, "o-", linewidth=1.6, markersize=4, color="C0")
            ax.set_xlabel(key)
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Parameter scan — {key}")
        fig.tight_layout()
    elif len(names) == 2:
        kx, ky = names
        xs = sorted(df[kx].unique())
        ys = sorted(df[ky].unique())
        grid = np.full((len(ys), len(xs)), np.nan)
        for _, r in df.iterrows():
            i = ys.index(r[ky])
            j = xs.index(r[kx])
            grid[i, j] = r["N_pc_hit"]

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{v:g}" for v in xs], rotation=45, ha="right")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{v:g}" for v in ys])
        ax.set_xlabel(kx)
        ax.set_ylabel(ky)
        ax.set_title(f"PC hits — {kx} × {ky}")
        for i in range(len(ys)):
            for j in range(len(xs)):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="w")
        fig.colorbar(im, ax=ax, label="PC hits")
        fig.tight_layout()
    else:
        print(f"[Scanner] 维度 {len(names)} > 2，跳过绘图。")
        return None

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Scanner] 图已保存 → {out_path}")
    return out_path


# =====================================================================
# CLI
# =====================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="XRF 通用参数扫描（基于 simulate，含 PC 窗口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----")[-1],
    )
    # ---- 晶体与几何基础参数（与 debug_tool 保持一致）----
    parser.add_argument("--crystal", type=str, default="flat",
                        choices=["flat", "cylindrical", "spiral"])
    parser.add_argument("--energy", type=float, default=183.0)
    parser.add_argument("--d_spacing", type=float, default=8.95e-9)
    parser.add_argument("--det_dist", type=float, default=124.0)
    parser.add_argument("--det_deg", type=float, default=30.0)
    parser.add_argument("--det_size", type=float, nargs=2, default=[22, 19])
    parser.add_argument("--cryst_dist", type=float, default=30.0)
    parser.add_argument("--cryst_size", type=float, nargs=2, default=[30, 40])
    parser.add_argument("--N", type=int, default=200_000)
    parser.add_argument("--fwhm", type=float, default=0.02)
    parser.add_argument("--spot_radius", type=float, default=20.0)
    parser.add_argument("--ROC", type=float, default=None)
    parser.add_argument("--a_crystal", type=float, default=None)
    parser.add_argument("--flatted", action="store_true")
    parser.add_argument("--pc_size", type=float, nargs=2, default=[30, 25])
    parser.add_argument("--cone", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（给定后每次模拟可复现）")
    # ---- 强度分布 ----
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--profile_x_range", type=float, nargs=2, default=None)
    parser.add_argument("--profile_y_range", type=float, nargs=2, default=None)
    # ---- 区域统计（可选）----
    parser.add_argument("--region_rect", type=float, nargs=4, action="append")
    parser.add_argument("--region_circle", type=float, nargs=3, action="append")
    # ---- 扫描控制 ----
    parser.add_argument("--sweep", type=str, action="append", required=False,
                        metavar="NAME=V1,V2,... | NAME=START:STOP:NUM",
                        help="扫描规则，可重复指定构成多维网格")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help=f"CSV / PNG 输出目录（默认 {DEFAULT_OUT_DIR}）")
    parser.add_argument("--tag", type=str, default=None,
                        help="输出文件名后缀，便于区分多组实验")
    parser.add_argument("--no_plot", action="store_true", help="不生成图片")
    parser.add_argument("--list-params", action="store_true",
                        help="列出 SimulationConfig 可扫描字段后退出")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.list_params:
        print("可扫描字段（SimulationConfig）：")
        for f in dataclasses.fields(SimulationConfig):
            print(f"  - {f.name:<22} default={f.default!r}")
        return

    if not args.sweep:
        raise SystemExit("[ERROR] 至少需要一条 --sweep，例如：--sweep \"fwhm_deg=0.01:0.2:10\"")

    sweeps = [parse_sweep(s) for s in args.sweep]
    names = [n for n, _ in sweeps]
    _validate_param_names(names)

    # 强度分布
    spot_profile = None
    if args.profile is not None:
        from xrf_sim.core.spot_profile import load_profile_file
        x_range = tuple(args.profile_x_range) if args.profile_x_range else None
        y_range = tuple(args.profile_y_range) if args.profile_y_range else None
        spot_profile = load_profile_file(args.profile, x_range, y_range)
        print(f"[Scanner] 已加载强度分布: {args.profile} shape={spot_profile[0].shape}")

    # 区域
    regions = []
    for x0, x1, y0, y1 in (args.region_rect or []):
        regions.append({"type": "rect", "x_range": (x0, x1), "y_range": (y0, y1)})
    for cx, cy, rad in (args.region_circle or []):
        regions.append({"type": "circle", "center": (cx, cy), "radius": rad})

    # 展开网格
    combos = [dict(zip(names, values))
              for values in itertools.product(*[v for _, v in sweeps])]

    # 输出路径
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"scan_{args.crystal}_{'_'.join(names)}{tag}"
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{stem}.csv")
    png_path = os.path.join(args.out_dir, f"{stem}.png")

    print(f"[Scanner] 晶体 = {args.crystal}  |  N = {args.N:,}  |  "
          f"组合数 = {len(combos)}")
    for n, v in sweeps:
        shown = ", ".join(f"{x:g}" for x in v[:8])
        more = " ..." if len(v) > 8 else ""
        print(f"          {n}: [{shown}{more}]")

    single_run = make_runner(args, spot_profile, regions, total=len(combos))
    df = run_scan(
        param_name=" x ".join(names),
        param_values=combos,
        run_func=single_run,
        csv_path=csv_path,
        verbose=False,
    )

    # 结果摘要
    show = names + ["N_hit_crystal", "N_reflected", "N_pc_hit", "pc_efficiency"]
    with_frac = [c for c in df.columns if c.startswith("pc_frac[")]
    show += with_frac
    print("\n[Scanner] 结果摘要：")
    print(df[show].to_string(index=False))

    if not args.no_plot:
        try:
            plot_scan(df, names, png_path)
        except Exception as e:      # 绘图失败不应破坏扫描结果
            print(f"[Scanner] WARN 绘图失败：{e}")

    return df


if __name__ == "__main__":
    main()
