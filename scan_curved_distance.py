"""
experiments/scripts/scan_curved_distance.py
============================================
约翰型弯晶通道：探测器距离扫描实验脚本。

对应原始 experiment.py 中弯晶距离扫描部分（detector_distance 从 80 到 200 mm）。

运行方式
--------
    python -m xrf_sim.experiments.scripts.scan_curved_distance
"""

import sys
import os
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from xrf_sim.core import WaferEmitter, BraggDiffraction
from xrf_sim.core.geometry import detector_geometry, bragg_angle_rad
from xrf_sim.experiments.scanner import run_scan

# ============================================================
# 实验参数配置
# ============================================================
ENERGY_CAL        = 183          # B K 线（eV）
CRYSTAL_D         = 8.95e-9      # 多层膜晶面间距（m）
CRYSTAL_ROC       = 150          # 弯晶曲率半径（mm）
DETECTOR_THETA_DEG = 30          # 探测器掠射角（度）
DETECTOR_SIZE     = (30, 5)      # 探测器入射面尺寸（mm）
CRYSTAL_SIZE      = (30, 42)     # 晶体有效尺寸（mm）
N_PHOTONS         = 10_000_000

CRYSTAL_THETA_DEG = np.degrees(bragg_angle_rad(ENERGY_CAL, CRYSTAL_D))
CRYSTAL_THETA_RAD = bragg_angle_rad(ENERGY_CAL, CRYSTAL_D)
print(f"■ crystal_theta_deg = {CRYSTAL_THETA_DEG:.4f}°")

spectrum = lambda n: ENERGY_CAL * np.ones(n)
emitter  = WaferEmitter(N_photons=N_PHOTONS, spectrum_sampler=spectrum, spot_radius=20.0)


def single_run(d_det: float) -> dict:
    """对给定探测器距离执行一次完整的模拟。"""
    crystal_distance = CRYSTAL_ROC * np.sin(CRYSTAL_THETA_RAD)

    bragg = BraggDiffraction(
        d_spacing=CRYSTAL_D,
        crystal_theta_deg=CRYSTAL_THETA_DEG,
        fwhm_deg=1.0,
        crystal_type="cylindrical",
        ROC=CRYSTAL_ROC,
    )

    photons, _ = emitter.emit(
        detector_distance=d_det,
        detector_theta_deg=DETECTOR_THETA_DEG,
        detector_size=DETECTOR_SIZE,
    )

    detector_normal, detector_center = detector_geometry(d_det, DETECTOR_THETA_DEG)

    reflected, hit_n = bragg.process(
        photons=photons,
        _detector_normal=detector_normal,
        _detector_center=detector_center,
        _crystal_distance=crystal_distance,
        _crystal_size=CRYSTAL_SIZE,
    )

    return {
        "detector_distance"  : d_det,
        "detector_theta_deg" : DETECTOR_THETA_DEG,
        "crystal_distance"   : crystal_distance,
        "crystal_ROC"        : CRYSTAL_ROC,
        "deg_fwhm"           : bragg.fwhm / np.pi * 180,
        "d_space"            : bragg.d,
        "bragg_angle"        : bragg.crystal_deg,
        "spot_radius"        : emitter.spot_radius,
        "total_photons"      : emitter.N,
        "slit_photons"       : len(photons),
        "hit_crystal"        : hit_n,
        "reflected_photons"  : len(reflected),
    }


if __name__ == "__main__":
    dist_list = np.linspace(80, 200, 13)

    run_scan(
        param_name="detector_distance",
        param_values=dist_list,
        run_func=single_run,
        csv_path=os.path.join(
            os.path.dirname(__file__), "../../results/scan_curved_B_distance.csv"
        ),
    )
