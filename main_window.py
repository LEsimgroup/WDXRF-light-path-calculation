"""
xrf_sim/ui/main_window.py
=========================
主窗口：左侧参数面板 + 右侧嵌入式 matplotlib 画布 + 底部统计。

模拟在 QThread 中执行，避免阻塞 UI；Live 模式下参数变化经防抖后触发重算。
"""

import numpy as np

from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QLabel, QStatusBar,
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from ..simulate import simulate, SimulationConfig
from ..plotting import draw_3d, draw_yz
from .controls import ControlsPanel


class _SimWorker(QObject):
    """在 worker 线程中执行 simulate。"""

    submit = Signal(object)
    finished = Signal(object)

    def __init__(self):
        super().__init__()
        # 跨线程排队连接
        self.submit.connect(self._run)

    @Slot(object)
    def _run(self, cfg: SimulationConfig):
        try:
            result = simulate(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[UI] simulate failed: {e}")
            result = None
        self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF 光路模拟器 — wafer → slit → 晶体 → PC 窗口")
        self.resize(1400, 850)

        self._busy = False
        self._pending = None      # 等待中的 config（busy 时暂存最新的）

        # --- 左侧控制面板 ---
        self.controls = ControlsPanel()
        self.controls.simulate_requested.connect(self._submit)

        # --- 右侧画布 ---
        self.figure = plt.figure(figsize=(9, 7))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.stats_label = QLabel("等待运行...")
        self.stats_label.setStyleSheet("padding:4px; background:#f5f5f5;")

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(2, 2, 2, 2)
        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas, 1)
        rv.addWidget(self.stats_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())

        # --- worker 线程 ---
        self._thread = QThread()
        self._worker = _SimWorker()
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._on_result)
        self._thread.start()

        # 初始绘制一次
        self._submit(self.controls.build_config(live=True))

    # ------------------------------------------------------------------
    # 提交模拟
    # ------------------------------------------------------------------
    def _submit(self, cfg: SimulationConfig):
        if self._busy:
            self._pending = cfg   # 暂存最新请求
            return
        self._busy = True
        self.statusBar().showMessage("模拟中…")
        self._worker.submit.emit(cfg)

    @Slot(object)
    def _on_result(self, result):
        self._busy = False
        if result is not None:
            self._render(result)
        # 处理暂存请求
        if self._pending is not None:
            cfg = self._pending
            self._pending = None
            self._submit(cfg)
        else:
            self.statusBar().showMessage("就绪")

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _render(self, result):
        view = self.controls.view
        self.figure.clear()
        np.random.seed(42)  # 可视化抽样稳定

        if view == "3D":
            ax = self.figure.add_subplot(111, projection="3d")
            draw_3d(ax, result, max_rays=300)
        elif view == "YZ":
            ax = self.figure.add_subplot(111)
            draw_yz(ax, result, max_rays=300)
        else:  # 3D + YZ
            ax3d = self.figure.add_subplot(1, 2, 1, projection="3d")
            draw_3d(ax3d, result, max_rays=200)
            ax_yz = self.figure.add_subplot(1, 2, 2)
            draw_yz(ax_yz, result, max_rays=200)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        cfg = result.config
        self.stats_label.setText(
            f"晶体: {cfg.crystal_type}   θ_B={result.bragg_angle_deg:.3f}°   "
            f"E={cfg.energy_eV} eV   |   "
            f"发射 {result.N_emitted:,} → slit {result.N_after_slit:,} → "
            f"命中晶体 {result.N_hit_crystal:,} → 反射 {result.N_reflected:,} → "
            f"<b style='color:#9C27B0'>PC 命中 {result.N_pc_hit:,}</b>   "
            f"效率 {result.efficiency:.4%}   PC效率 {result.pc_efficiency:.4%}"
        )

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)
