"""
experiments/scripts/scan_flat_angle.py
=======================================
平晶通道：探测器角度扫描实验脚本。

对应原始 experiment.py 中最后一段活跃代码（平晶 B 元素 K 线，
detector_theta_deg 从 25° 到 45° 遍历）。

运行方式
--------
在项目根目录（xrf_sim 的上一级）执行：

    python -m xrf_sim.experiments.scripts.scan_flat_angle

或直接：

    python experiments/scripts/scan_flat_angle.py
"""

import sys
import os
import numpy as np

# 确保项目根目录在 sys.path 中（直接运行脚本时需要）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xrf_sim.core import WaferEmitter, BraggDiffraction
from xrf_sim.core.geometry import detector_geometry, bragg_angle_rad
from xrf_sim.experiments.scanner import run_scan

# ============================================================
# 实验参数配置
# ============================================================
H = 4.135667696e-15   # eV·s
C = 299792458         # m/s

ENERGY_CAL    = 183          # B K 线（eV）
CRYSTAL_D     = 8.95e-9      # 多层膜平晶晶面间距（m）
CRYSTAL_DIST  = 90           # 晶体到探测器的距离（mm）
DETECTOR_DIST = 124          # 探测器到 wafer 的距离（mm）
DETECTOR_SIZE = (25, 21)     # 探测器入射面尺寸 (width, height)，mm
CRYSTAL_SIZE  = (30, 50)     # 晶体有效尺寸 (width, height)，mm
N_PHOTONS     = 10_000_000   # 总光子数

# 布拉格角（由晶面间距和光子能量计算）
CRYSTAL_THETA_DEG = np.degrees(bragg_angle_rad(ENERGY_CAL, CRYSTAL_D))
print(f"■ crystal_theta_deg = {CRYSTAL_THETA_DEG:.4f}°")

# 能量谱（单色光）
spectrum = lambda n: ENERGY_CAL * np.ones(n)

# 光源
emitter = WaferEmitter(
    N_photons=N_PHOTONS,
    spectrum_sampler=spectrum,
    spot_radius=20.0,
)

# ============================================================
# 单次模拟函数
# ============================================================

def single_run(detector_theta_deg: float) -> dict:
    """对给定探测器角度执行一次完整的模拟，返回结果字典。"""

    # 构建 Bragg 衍射器（平晶，crystal_type='flat'）
    bragg = BraggDiffraction(
        d_spacing=CRYSTAL_D,
        crystal_theta_deg=CRYSTAL_THETA_DEG,
        fwhm_deg=1.0,
        crystal_type="flat",
    )

    # 发射光子并筛选通过探测器入射面的光子
    photons, _ = emitter.emit(
        detector_distance=DETECTOR_DIST,
        detector_theta_deg=detector_theta_deg,
        detector_size=DETECTOR_SIZE,
    )

    # 计算探测器几何（与 emit 内部保持一致）
    detector_normal, detector_center = detector_geometry(
        DETECTOR_DIST, detector_theta_deg
    )

    # 模拟晶体反射
    reflected, hit_n = bragg.process(
        photons=photons,
        _detector_normal=detector_normal,
        _detector_center=detector_center,
        _crystal_distance=CRYSTAL_DIST,
        _crystal_size=CRYSTAL_SIZE,
    )

    return {
        "detector_distance"  : DETECTOR_DIST,
        "detector_theta_deg" : detector_theta_deg,
        "crystal_distance"   : CRYSTAL_DIST,
        "crystal_ROC"        : None,
        "deg_fwhm"           : bragg.fwhm / np.pi * 180,
        "d_space"            : bragg.d,
        "bragg_angle"        : bragg.crystal_deg,
        "spot_radius"        : emitter.spot_radius,
        "total_photons"      : emitter.N,
        "slit_photons"       : len(photons),
        "hit_crystal"        : hit_n,
        "reflected_photons"  : len(reflected),
    }


# ============================================================
# 执行扫描
# ============================================================
if __name__ == "__main__":
    angle_list = np.linspace(25, 45, 21)

    run_scan(
        param_name="detector_theta_deg",
        param_values=angle_list,
        run_func=single_run,
        csv_path=os.path.join(
            os.path.dirname(__file__), "../../results/scan_flat_B_angle.csv"
        ),
    )
