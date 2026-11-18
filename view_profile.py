#!/usr/bin/env python3
"""
view_profile.py
===============
独立的强度分布文件查看工具。

用于加载并可视化二维 XY 强度分布文件（.npy / .csv / .txt），
不运行光路模拟。可查看：
    - 强度分布热力图
    - 分布统计信息（最大值、中心、范围）
    - 沿 x / y 的强度剖面

使用方式
--------
命令行::

    python -m xrf_sim.view_profile --file ./sim_output/test_profile.npy
    python -m xrf_sim.view_profile --file ./intensity.csv --x_range -25 25 --y_range -25 25
    python -m xrf_sim.view_profile --file ./data.npy --save_dir ./sim_output
"""

import os
import sys
import numpy as np


def load_profile(filepath: str):
    """加载强度分布文件，返回 (profile, x_range, y_range)。"""
    from .core.spot_profile import load_profile_file
    return load_profile_file(filepath)


def view_profile(
    filepath: str,
    x_range: tuple = None,
    y_range: tuple = None,
    save_dir: str = None,
) -> dict:
    """
    加载并可视化强度分布。

    返回 dict 包含 profile、范围与统计信息。
    """
    profile, xr, yr = load_profile(filepath)
    if x_range is not None:
        xr = tuple(x_range)
    if y_range is not None:
        yr = tuple(y_range)

    Ny, Nx = profile.shape
    total = float(profile.sum())
    peak = float(profile.max())
    mean = float(profile.mean())

    print("=" * 60)
    print(f"  强度分布文件: {filepath}")
    print("=" * 60)
    print(f"  shape        : {profile.shape}  (Ny={Ny}, Nx={Nx})")
    print(f"  x 范围       : {xr}  mm")
    print(f"  y 范围       : {yr}  mm")
    print(f"  峰值强度     : {peak:.6g}")
    print(f"  平均强度     : {mean:.6g}")
    print(f"  总强度       : {total:.6g}")
    print(f"  最小值       : {float(profile.min()):.6g}")
    print("=" * 60)

    # 可视化
    import matplotlib
    matplotlib.use("Agg")  # 非交互，直接保存
    import matplotlib.pyplot as plt

    x = np.linspace(xr[0], xr[1], Nx)
    y = np.linspace(yr[0], yr[1], Ny)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 强度热图
    im = axes[0, 0].imshow(
        profile, extent=[xr[0], xr[1], yr[0], yr[1]],
        origin="lower", aspect="auto", cmap="viridis"
    )
    axes[0, 0].set_title("Intensity Map (XY)")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=axes[0, 0], label="Intensity")

    # 2. 沿 x 轴剖面（在 y 峰值处）
    peak_y_idx = int(np.argmax(profile.max(axis=1)))
    axes[0, 1].plot(x, profile[peak_y_idx, :])
    axes[0, 1].set_title(f"Profile along x (y={y[peak_y_idx]:.1f} mm)")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("Intensity")

    # 3. 沿 y 轴剖面（在 x 峰值处）
    peak_x_idx = int(np.argmax(profile.max(axis=0)))
    axes[1, 0].plot(y, profile[:, peak_x_idx])
    axes[1, 0].set_title(f"Profile along y (x={x[peak_x_idx]:.1f} mm)")
    axes[1, 0].set_xlabel("y (mm)")
    axes[1, 0].set_ylabel("Intensity")

    # 4. 强度直方图
    axes[1, 1].hist(profile.ravel(), bins=50)
    axes[1, 1].set_title("Intensity histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "profile_view.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[view_profile] 已保存: {out}")
    else:
        out = None
        print("[view_profile] 未指定 --save_dir，未保存图片。")

    plt.close(fig)

    return {
        "profile": profile,
        "x_range": xr,
        "y_range": yr,
        "peak": peak,
        "mean": mean,
        "total": total,
        "figure_path": out,
    }


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="查看二维强度分布文件（不运行光路模拟）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", type=str, required=True,
                        help="强度分布文件路径 (.npy/.csv/.txt)")
    parser.add_argument("--x", type=float, nargs=2, default=None,
                        help="x 范围 xmin xmax mm (default: -25 25)")
    parser.add_argument("--y", type=float, nargs=2, default=None,
                        help="y 范围 ymin ymax mm (default: -25 25)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="图片保存目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    view_profile(
        filepath=args.file,
        x_range=tuple(args.x) if args.x else None,
        y_range=tuple(args.y) if args.y else None,
        save_dir=args.save_dir,
    )