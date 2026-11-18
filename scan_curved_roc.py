"""
experiments/scripts/scan_curved_roc.py
=======================================
约翰型弯晶通道：曲率半径（ROC）扫描实验脚本。

对应原始 experiment.py 中弯晶 ROC 扫描部分（crystal_ROC 从 80 到 300 mm）。

运行方式
--------
    python -m xrf_sim.experiments.scripts.scan_curved_roc
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
ENERGY_CAL         = 183
CRYSTAL_D          = 8.95e-9
DETECTOR_DIST      = 100          # mm
DETECTOR_THETA_DEG = 30
DETECTOR_SIZE      = (30, 5)
CRYSTAL_SIZE       = (30, 42)
N_PHOTONS          = 10_000_000

CRYSTAL_THETA_DEG = np.degrees(bragg_angle_rad(ENERGY_CAL, CRYSTAL_D))
CRYSTAL_THETA_RAD = bragg_angle_rad(ENERGY_CAL, CRYSTAL_D)
print(f"■ crystal_theta_deg = {CRYSTAL_THETA_DEG:.4f}°")

spectrum = lambda n: ENERGY_CAL * np.ones(n)
emitter  = WaferEmitter(N_photons=N_PHOTONS, spectrum_sampler=spectrum, spot_radius=20.0)


def single_run(crystal_ROC: float) -> dict:
    """对给定曲率半径执行一次完整的模拟。"""
    crystal_distance = crystal_ROC * np.sin(CRYSTAL_THETA_RAD)

    bragg = BraggDiffraction(
        d_spacing=CRYSTAL_D,
        crystal_theta_deg=CRYSTAL_THETA_DEG,
        fwhm_deg=0.1,
        crystal_type="cylindrical",
        ROC=crystal_ROC,
    )

    photons, _ = emitter.emit(
        detector_distance=DETECTOR_DIST,
        detector_theta_deg=DETECTOR_THETA_DEG,
        detector_size=DETECTOR_SIZE,
    )

    detector_normal, detector_center = detector_geometry(
        DETECTOR_DIST, DETECTOR_THETA_DEG
    )

    reflected, hit_n = bragg.process(
        photons=photons,
        _detector_normal=detector_normal,
        _detector_center=detector_center,
        _crystal_distance=crystal_distance,
        _crystal_size=CRYSTAL_SIZE,
    )

    return {
        "detector_distance"  : DETECTOR_DIST,
        "detector_theta_deg" : DETECTOR_THETA_DEG,
        "crystal_distance"   : crystal_distance,
        "crystal_ROC"        : crystal_ROC,
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
    roc_list = np.linspace(80, 300, 23)

    run_scan(
        param_name="crystal_ROC",
        param_values=roc_list,
        run_func=single_run,
        csv_path=os.path.join(
            os.path.dirname(__file__), "../../results/scan_curved_B_roc.csv"
        ),
    )
