"""
xrf_sim/ui/controls.py
======================
参数控制面板。集中所有光路调节参数，提供 Live 实时模式与 Run 按钮。
任一参数变化时，经防抖后发出 ``simulate_requested`` 信号。
"""

from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QScrollArea, QLabel,
)

from ..simulate import SimulationConfig


def _dspin(default, mn, mx, step, decimals=4, suffix=""):
    sb = QDoubleSpinBox()
    sb.setRange(mn, mx)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(default)
    if suffix:
        sb.setSuffix(suffix)
    return sb


class ControlsPanel(QWidget):
    """左侧参数控制面板。"""

    simulate_requested = Signal(object)   # 发出 SimulationConfig

    CRYSTAL_TYPES = ["flat", "cylindrical", "spiral"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._emit_config)
        # 晶体类型变化时立即刷新控件启用状态
        self.crystal_type.currentTextChanged.connect(self._update_enabled)
        self._update_enabled()

    # ------------------------------------------------------------------
    # UI 构造
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setLabelAlignment(form.labelAlignment())

        # 晶体类型
        self.crystal_type = QComboBox()
        self.crystal_type.addItems(self.CRYSTAL_TYPES)
        form.addRow("晶体类型", self.crystal_type)

        # 光源 / 能量
        self.energy = _dspin(183.0, 10, 1e5, 1, decimals=2, suffix=" eV")
        self.d_spacing = _dspin(8.95e-9, 1e-12, 1e-6, 1e-10, decimals=12, suffix=" m")
        self.N_photons = QSpinBox()
        self.N_photons.setRange(1000, 5_000_000)
        self.N_photons.setSingleStep(10000)
        self.N_photons.setValue(100_000)
        self.spot_radius = _dspin(20.0, 0.1, 200, 1, decimals=2, suffix=" mm")
        self.fwhm = _dspin(0.02, 1e-4, 10, 0.01, decimals=4, suffix=" °")
        form.addRow("光子能量", self.energy)
        form.addRow("晶面间距 d", self.d_spacing)
        form.addRow("光子数 N", self.N_photons)
        form.addRow("光斑半径", self.spot_radius)
        form.addRow("反射率 FWHM", self.fwhm)

        # 探测器 slit
        self.det_dist = _dspin(124.0, 10, 1000, 1, decimals=2, suffix=" mm")
        self.det_deg = _dspin(30.0, 1, 89, 0.5, decimals=2, suffix=" °")
        self.det_w = _dspin(25.0, 1, 500, 1, decimals=2)
        self.det_h = _dspin(5.0, 1, 500, 1, decimals=2)
        form.addRow("slit 距离", self.det_dist)
        form.addRow("slit 掠射角", self.det_deg)
        form.addRow("slit 宽 / 高", self._pair_label(self.det_w, self.det_h))

        # 晶体
        self.cryst_dist = _dspin(30.0, 1, 500, 1, decimals=2, suffix=" mm")
        self.cryst_w = _dspin(30.0, 1, 500, 1, decimals=2)
        self.cryst_h = _dspin(40.0, 1, 500, 1, decimals=2)
        self.ROC = _dspin(150.0, 10, 2000, 5, decimals=2, suffix=" mm")
        self.a_crystal = _dspin(60.0, 5, 500, 1, decimals=2, suffix=" mm")
        form.addRow("晶体距离", self.cryst_dist)
        form.addRow("晶体宽 / 弧长", self._pair_label(self.cryst_w, self.cryst_h))
        form.addRow("ROC (cyl)", self.ROC)
        form.addRow("a (spiral)", self.a_crystal)

        # PC 窗口
        self.pc_w = _dspin(25.0, 1, 500, 1, decimals=2)
        self.pc_h = _dspin(5.0, 1, 500, 1, decimals=2)
        form.addRow("PC 窗口宽 / 高", self._pair_label(self.pc_w, self.pc_h))

        # 视图
        self.view_mode = QComboBox()
        self.view_mode.addItems(["3D", "YZ", "3D + YZ"])
        form.addRow("视图", self.view_mode)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # 运行控制
        self.live_chk = QCheckBox("Live 实时更新（小 N）")
        self.live_chk.setChecked(True)
        self.live_n = QSpinBox()
        self.live_n.setRange(1000, 200_000)
        self.live_n.setSingleStep(5000)
        self.live_n.setValue(30_000)
        self.live_n.setSuffix(" photons")
        self.run_btn = QPushButton("Run（完整模拟）")
        outer.addWidget(self.live_chk)
        outer.addWidget(self.live_n)
        outer.addWidget(self.run_btn)

        # 连接变化信号 → 防抖
        for w in self._all_spinboxes():
            w.valueChanged.connect(self._kick)
        self.crystal_type.currentTextChanged.connect(self._kick)
        self.view_mode.currentTextChanged.connect(self._kick)
        self.live_n.valueChanged.connect(self._kick)
        self.run_btn.clicked.connect(self._emit_config_full)

    def _pair_label(self, w1, w2):
        wrap = QWidget()
        h = QVBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w1)
        h.addWidget(w2)
        return wrap

    def _all_spinboxes(self):
        return [
            self.energy, self.d_spacing, self.spot_radius, self.fwhm,
            self.det_dist, self.det_deg, self.det_w, self.det_h,
            self.cryst_dist, self.cryst_w, self.cryst_h, self.ROC, self.a_crystal,
            self.pc_w, self.pc_h,
        ]

    # ------------------------------------------------------------------
    # 行为
    # ------------------------------------------------------------------
    def _update_enabled(self):
        ct = self.crystal_type.currentText()
        self.ROC.setEnabled(ct == "cylindrical")
        self.a_crystal.setEnabled(ct == "spiral")

    def _kick(self):
        """任意参数变化 → 重启防抖定时器。"""
        if self.live_chk.isChecked():
            self._debounce.start()

    def _emit_config(self):
        self.simulate_requested.emit(self.build_config(live=True))

    def _emit_config_full(self):
        self.simulate_requested.emit(self.build_config(live=False))

    def build_config(self, live: bool) -> SimulationConfig:
        ct = self.crystal_type.currentText()
        n = self.live_n.value() if live else self.N_photons.value()
        return SimulationConfig(
            crystal_type=ct,
            energy_eV=self.energy.value(),
            d_spacing=self.d_spacing.value(),
            detector_distance=self.det_dist.value(),
            detector_theta_deg=self.det_deg.value(),
            detector_size=(self.det_w.value(), self.det_h.value()),
            crystal_distance=self.cryst_dist.value(),
            crystal_size=(self.cryst_w.value(), self.cryst_h.value()),
            N_photons=n,
            fwhm_deg=self.fwhm.value(),
            spot_radius=self.spot_radius.value(),
            ROC=self.ROC.value() if ct == "cylindrical" else None,
            a_crystal=self.a_crystal.value() if ct == "spiral" else None,
            pc_window_size=(self.pc_w.value(), self.pc_h.value()),
            verbose=False,
        )

    @property
    def view(self) -> str:
        return self.view_mode.currentText()
