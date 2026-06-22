"""GUI desktop (PyQt6 + PyVista): panel parameter + render live + export MP4."""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QRadioButton, QButtonGroup, QSplitter, QFrame,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

import pyvista as pv
from pyvistaqt import QtInteractor

from .config import SimulationConfig
from .solver import NavierStokesSolver
from . import backend

OBSTACLES = ["Cylinder", "Ellipse", "Square", "Diamond", "Hexagon", "Triangle", "Flat Plate"]
OBS_MAP = {0: "cylinder", 1: "ellipse", 2: "square", 3: "diamond",
           4: "hexagon", 5: "triangle", 6: "plate"}


class SimulationWorker(QThread):
    # step, time, max|div|, p_resid, langkah/detik
    step_done = pyqtSignal(int, float, float, float, float)
    finished = pyqtSignal()

    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self._running = True

    def run(self):
        cfg = self.solver.cfg
        t_last = time.perf_counter()
        n_last = self.solver.step_count
        while self._running and self.solver.step_count < cfg.n_steps:
            self.solver.step()
            if self.solver.step_count % cfg.plot_every == 0:
                self.solver.update_diagnostics()
                now = time.perf_counter()
                dt = now - t_last
                sps = (self.solver.step_count - n_last) / dt if dt > 0 else 0.0
                t_last, n_last = now, self.solver.step_count
                self.step_done.emit(self.solver.step_count, self.solver.time,
                                    self.solver.divergence_error,
                                    self.solver.poisson_residual, sps)
        self.finished.emit()

    def stop(self):
        self._running = False


class ExportWorker(QThread):
    # render MP4 di latar belakang (solver terpisah, matplotlib Agg)
    progress = pyqtSignal(float)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg, field, out_path, n_frames, frame_every, warmup, fps, mode):
        super().__init__()
        self.cfg, self.field, self.out_path = cfg, field, out_path
        self.n_frames, self.frame_every = n_frames, frame_every
        self.warmup, self.fps, self.mode = warmup, fps, mode

    def run(self):
        try:
            from .render import render_mp4
            render_mp4(self.cfg, field=self.field, out_path=self.out_path,
                       n_frames=self.n_frames, frame_every=self.frame_every,
                       warmup=self.warmup, fps=self.fps, mode=self.mode,
                       progress=lambda f: self.progress.emit(float(f)))
            self.done.emit(self.out_path)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navier-Stokes 2D Solver  |  Vortex Shedding & Heat Transfer")
        self.resize(1440, 820)
        self._apply_theme()

        self.solver = None
        self.worker = None
        self.export_worker = None

        # state render
        self._grid = None
        self._actor = None
        self._cmap = None
        self._dims = None
        self._vort_clim = None
        self._last_render = None

        # render dipisah dari simulasi: timer 60 Hz menggambar frame terbaru,
        # solver melaju bebas di worker thread -> animasi mulus s/d 60 FPS
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(16)      # ~60 FPS
        self._render_timer.timeout.connect(self._on_render_tick)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())

        self.pv_frame = QFrame()
        pv_layout = QVBoxLayout(self.pv_frame)
        pv_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.pv_frame)
        pv_layout.addWidget(self.plotter.interactor)
        splitter.addWidget(self.pv_frame)

        splitter.setSizes([300, 1140])
        main_layout.addWidget(splitter)
        self._init_pyvista()

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        hdr = QLabel("SIMULATION PARAMETERS")
        hdr.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet("color: #89b4fa; padding: 4px 0;")
        layout.addWidget(hdr)

        # geometri + orientasi penghalang
        grp1 = QGroupBox("Obstacle Geometry")
        l1 = QVBoxLayout(grp1)
        self.combo_obstacle = QComboBox()
        self.combo_obstacle.addItems(OBSTACLES)
        l1.addWidget(self.combo_obstacle)

        row = QHBoxLayout()
        row.addWidget(QLabel("Size (D)"))
        self.spin_size = QDoubleSpinBox()
        self.spin_size.setRange(0.3, 5.0)
        self.spin_size.setValue(1.0)
        self.spin_size.setSingleStep(0.25)
        row.addWidget(self.spin_size)
        row.addWidget(QLabel("Angle"))
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-90, 90)
        self.spin_angle.setValue(0)
        self.spin_angle.setSingleStep(5)
        self.spin_angle.setSuffix(" °")
        row.addWidget(self.spin_angle)
        l1.addLayout(row)

        self.lbl_area = QLabel("luas ≈ --")
        self.lbl_area.setStyleSheet("color:#94e2d5; font-family:Consolas; font-size:10px;")
        l1.addWidget(self.lbl_area)
        layout.addWidget(grp1)

        # update readout luas saat geometri berubah
        for w in (self.combo_obstacle,):
            w.currentIndexChanged.connect(self._update_area)
        for w in (self.spin_size, self.spin_angle):
            w.valueChanged.connect(self._update_area)

        grp2 = QGroupBox("Reynolds Number")
        l2 = QVBoxLayout(grp2)
        self.spin_re = QSpinBox()
        self.spin_re.setRange(20, 1000)
        self.spin_re.setValue(150)
        self.spin_re.setSingleStep(10)
        l2.addWidget(self.spin_re)
        layout.addWidget(grp2)

        grp3 = QGroupBox("Grid Resolution")
        l3 = QHBoxLayout(grp3)
        self.spin_nx = QSpinBox()
        self.spin_nx.setRange(50, 800)
        self.spin_nx.setValue(300)
        self.spin_nx.setSingleStep(50)
        l3.addWidget(QLabel("Nx"))
        l3.addWidget(self.spin_nx)
        self.spin_ny = QSpinBox()
        self.spin_ny.setRange(20, 400)
        self.spin_ny.setValue(120)
        self.spin_ny.setSingleStep(10)
        l3.addWidget(QLabel("Ny"))
        l3.addWidget(self.spin_ny)
        layout.addWidget(grp3)

        grp4 = QGroupBox("Advection Discretization")
        l4 = QVBoxLayout(grp4)
        self.radio_fdm = QRadioButton("FDM  (Central Difference, O2)")
        self.radio_fvm = QRadioButton("FVM  (Blended Flux)")
        self.radio_fvm.setChecked(True)
        self.bg_method = QButtonGroup()
        self.bg_method.addButton(self.radio_fdm)
        self.bg_method.addButton(self.radio_fvm)
        l4.addWidget(self.radio_fdm)
        l4.addWidget(self.radio_fvm)
        layout.addWidget(grp4)

        grpb = QGroupBox("Compute Backend")
        lb = QVBoxLayout(grpb)
        self.radio_cpu = QRadioButton("CPU  (Numba, multicore)")
        gpu_txt = "GPU  (CuPy)" if backend.GPU_PRESENT else "GPU  (tidak tersedia)"
        self.radio_gpu = QRadioButton(gpu_txt)
        self.radio_gpu.setEnabled(backend.GPU_PRESENT)
        self.bg_backend = QButtonGroup()
        self.bg_backend.addButton(self.radio_cpu)
        self.bg_backend.addButton(self.radio_gpu)
        if backend.default_mode() == "gpu":
            self.radio_gpu.setChecked(True)
        else:
            self.radio_cpu.setChecked(True)
        lb.addWidget(self.radio_cpu)
        lb.addWidget(self.radio_gpu)
        prow = QHBoxLayout()
        self.radio_fp32 = QRadioButton("fp32")
        self.radio_fp64 = QRadioButton("fp64")
        self.radio_fp32.setChecked(True)
        self.bg_prec = QButtonGroup()
        self.bg_prec.addButton(self.radio_fp32)
        self.bg_prec.addButton(self.radio_fp64)
        prow.addWidget(QLabel("Precision"))
        prow.addWidget(self.radio_fp32)
        prow.addWidget(self.radio_fp64)
        lb.addLayout(prow)
        layout.addWidget(grpb)

        grp5 = QGroupBox("Field Display")
        l5 = QVBoxLayout(grp5)
        self.radio_vort = QRadioButton("Vorticity")
        self.radio_temp = QRadioButton("Temperature")
        self.radio_vel = QRadioButton("Velocity Magnitude")
        self.radio_vort.setChecked(True)
        self.bg_vis = QButtonGroup()
        for r in (self.radio_vort, self.radio_temp, self.radio_vel):
            self.bg_vis.addButton(r)
            l5.addWidget(r)
        layout.addWidget(grp5)
        self.radio_vort.toggled.connect(self._on_vis_changed)
        self.radio_temp.toggled.connect(self._on_vis_changed)
        self.radio_vel.toggled.connect(self._on_vis_changed)

        grp6 = QGroupBox("Control")
        l6 = QVBoxLayout(grp6)
        self.btn_start = QPushButton("START")
        self.btn_start.setStyleSheet(self._btn_style("#2ecc71", "#27ae60"))
        self.btn_start.clicked.connect(self._on_start)
        l6.addWidget(self.btn_start)
        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet(self._btn_style("#f39c12", "#e67e22"))
        self.btn_pause.clicked.connect(self._on_pause)
        l6.addWidget(self.btn_pause)
        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setStyleSheet(self._btn_style("#e74c3c", "#c0392b"))
        self.btn_reset.clicked.connect(self._on_reset)
        l6.addWidget(self.btn_reset)
        self.btn_export = QPushButton("EXPORT MP4")
        self.btn_export.setToolTip("Render field terpilih ke Hasil/*.mp4 (30 FPS, latar belakang)")
        self.btn_export.setStyleSheet(self._btn_style("#89b4fa", "#74a0f0"))
        self.btn_export.clicked.connect(self._on_export)
        l6.addWidget(self.btn_export)
        self.lbl_export = QLabel("")
        self.lbl_export.setStyleSheet("color:#a6e3a1; font-family:Consolas; font-size:10px;")
        self.lbl_export.setWordWrap(True)
        l6.addWidget(self.lbl_export)
        layout.addWidget(grp6)

        grp7 = QGroupBox("Diagnostics")
        l7 = QVBoxLayout(grp7)
        mono = QFont("Consolas", 10)
        self.lbl_step = QLabel("step     : 0")
        self.lbl_time = QLabel("time     : 0.0000")
        self.lbl_dt = QLabel("dt       : --")
        self.lbl_div = QLabel("max|div| : --")
        self.lbl_pres = QLabel("p_resid  : --")
        self.lbl_sps = QLabel("sim sps  : --")
        self.lbl_fps = QLabel("render fps: --")
        self.lbl_gpu = QLabel("backend  : " + backend.backend_label(backend.default_mode()))
        for lbl in (self.lbl_step, self.lbl_time, self.lbl_dt, self.lbl_div,
                    self.lbl_pres, self.lbl_sps, self.lbl_fps, self.lbl_gpu):
            lbl.setFont(mono)
            l7.addWidget(lbl)
        layout.addWidget(grp7)

        self.spin_nx.valueChanged.connect(self._update_area)
        self.spin_ny.valueChanged.connect(self._update_area)
        self._update_area()

        layout.addStretch()
        return panel

    @staticmethod
    def _btn_style(c, ch):
        return (f"QPushButton {{ background-color: {c}; color: #1e1e2e; "
                "font-weight: bold; padding: 8px; border-radius: 4px; "
                "font-family: Consolas; font-size: 13px; }"
                f"QPushButton:hover {{ background-color: {ch}; }}"
                "QPushButton:disabled { background-color: #45475a; color: #6c7086; }")

    # --- PyVista ---

    def _init_pyvista(self):
        self.plotter.set_background("#181825")
        self.plotter.clear()
        self._grid = self._actor = self._cmap = self._dims = None
        grid = pv.ImageData(dimensions=(2, 2, 1))
        grid.point_data["values"] = np.zeros(4)
        self.plotter.add_mesh(grid, scalars="values", cmap="RdBu_r",
                              show_scalar_bar=False, lighting=False)
        self.plotter.view_xy()

    def _current_field(self):
        cfg = self.solver.cfg
        if self.radio_vort.isChecked():
            data = self.solver.get_vorticity()
            absmax = float(np.percentile(np.abs(data), 99.0))
            if self._vort_clim is None and absmax > 1e-6:
                self._vort_clim = (-absmax, absmax)        # bekukan skala warna
            clim = self._vort_clim or (-max(absmax, 1e-9), max(absmax, 1e-9))
            return data, "RdBu_r", "Vorticity", clim
        if self.radio_temp.isChecked():
            data = self.solver.get_temperature()
            return data, "inferno", "Temperature", (cfg.T_inf, cfg.T_obs)
        data = self.solver.get_velocity_magnitude()
        return data, "viridis", "Velocity", (float(data.min()), max(float(data.max()), 1e-9))

    def _update_pyvista(self):
        if self.solver is None:
            return
        cfg = self.solver.cfg
        data, cmap, title, clim = self._current_field()

        data = data.astype(np.float64, copy=True)
        data[self.solver.get_obstacle_mask()] = np.nan      # siluet obstacle
        flat = np.ascontiguousarray(data.ravel(order="C"))

        dims = (cfg.nx + 1, cfg.ny + 1, 1)
        if self._grid is None or self._cmap != cmap or self._dims != dims:
            self._rebuild_mesh(flat, cmap, title, clim, dims, cfg)
        else:
            self._grid.cell_data["values"][:] = flat
            self._actor.mapper.scalar_range = clim
        self.plotter.render()

        now = time.perf_counter()
        if self._last_render is not None:
            dt = now - self._last_render
            if dt > 0:
                self.lbl_fps.setText(f"render fps: {1.0/dt:6.1f}")
        self._last_render = now

    def _rebuild_mesh(self, flat, cmap, title, clim, dims, cfg):
        self.plotter.clear()
        grid = pv.ImageData(dimensions=dims, spacing=(cfg.dx, cfg.dy, 1.0))
        grid.cell_data["values"] = flat
        self._actor = self.plotter.add_mesh(
            grid, scalars="values", cmap=cmap, clim=clim,
            interpolate_before_map=True, lighting=False,
            nan_color="#9399b2", nan_opacity=1.0,
            scalar_bar_args={"title": title, "color": "white",
                             "title_font_size": 13, "label_font_size": 11})
        self._grid = grid
        self._cmap = cmap
        self._dims = dims
        self.plotter.view_xy()

    def _on_vis_changed(self):
        self._grid = None       # cmap/clim berubah -> bangun ulang mesh
        self._update_pyvista()

    # --- aksi ---

    def _selected_mode(self):
        return "gpu" if self.radio_gpu.isChecked() else "cpu"

    def _on_start(self):
        if self.solver is None:
            cfg = self._build_config()
            self.solver = NavierStokesSolver(cfg, self._selected_mode())
            self.lbl_dt.setText(f"dt       : {self.solver.dt:.6f}")
            self.lbl_gpu.setText(f"backend  : {backend.backend_label(self.solver.mode)}")
            self._vort_clim = None
            self._grid = None
            self._update_pyvista()
        self.worker = SimulationWorker(self.solver)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._render_timer.start()          # render 60 FPS, lepas dari sim
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self._set_inputs_enabled(False)

    def _on_pause(self):
        self._render_timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def _on_reset(self):
        self._render_timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.solver = None
        self._vort_clim = None
        self._last_render = None
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self._set_inputs_enabled(True)
        for lbl, txt in [(self.lbl_step, "step     : 0"), (self.lbl_time, "time     : 0.0000"),
                         (self.lbl_dt, "dt       : --"), (self.lbl_div, "max|div| : --"),
                         (self.lbl_pres, "p_resid  : --"), (self.lbl_sps, "sim sps  : --"),
                         (self.lbl_fps, "render fps: --")]:
            lbl.setText(txt)
        self._init_pyvista()

    def _on_step_done(self, step, t, div_err, p_res, sps):
        # hanya update label diagnostik; render ditangani timer 60 FPS
        self.lbl_step.setText(f"step     : {step:,}")
        self.lbl_time.setText(f"time     : {t:.4f}")
        self.lbl_div.setText(f"max|div| : {div_err:.2e}")
        self.lbl_pres.setText(f"p_resid  : {p_res:.2e}")
        self.lbl_sps.setText(f"sim sps  : {sps:7.1f}")

    def _on_render_tick(self):
        if self.solver is not None:
            self._update_pyvista()

    def _on_finished(self):
        self._render_timer.stop()
        self._update_pyvista()              # gambar frame terakhir
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)

    # --- export MP4 ---

    def _selected_field_name(self):
        if self.radio_temp.isChecked():
            return "temperature"
        if self.radio_vel.isChecked():
            return "velocity"
        return "vorticity"

    def _on_export(self):
        if self.export_worker is not None:
            return
        cfg = self._build_config()
        field = self._selected_field_name()
        out = f"Hasil/{cfg.obstacle_type}_{field}_Re{int(cfg.Re)}.mp4"
        self.export_worker = ExportWorker(cfg, field, out, n_frames=300,
                                          frame_every=15, warmup=6000, fps=30,
                                          mode=self._selected_mode())
        self.export_worker.progress.connect(
            lambda f: self.lbl_export.setText(f"export {field}: {f*100:4.0f}%"))
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.start()
        self.btn_export.setEnabled(False)
        self.lbl_export.setStyleSheet("color:#f9e2af; font-family:Consolas; font-size:10px;")
        self.lbl_export.setText(f"export {field}: 0%")

    def _on_export_done(self, path):
        self.export_worker = None
        self.btn_export.setEnabled(True)
        self.lbl_export.setStyleSheet("color:#a6e3a1; font-family:Consolas; font-size:10px;")
        self.lbl_export.setText(f"tersimpan: {path}")

    def _on_export_failed(self, msg):
        self.export_worker = None
        self.btn_export.setEnabled(True)
        self.lbl_export.setStyleSheet("color:#f38ba8; font-family:Consolas; font-size:10px;")
        self.lbl_export.setText(f"gagal: {msg}")

    # --- utility ---

    def _build_config(self):
        method = "fdm" if self.radio_fdm.isChecked() else "fvm"
        precision = "single" if self.radio_fp32.isChecked() else "double"
        return SimulationConfig(
            nx=self.spin_nx.value(), ny=self.spin_ny.value(),
            Re=float(self.spin_re.value()),
            obstacle_type=OBS_MAP.get(self.combo_obstacle.currentIndex(), "cylinder"),
            obs_D=self.spin_size.value(), obs_angle=float(self.spin_angle.value()),
            method=method, precision=precision)

    def _update_area(self):
        from .grid import obstacle_area
        cfg = SimulationConfig(
            nx=self.spin_nx.value(), ny=self.spin_ny.value(),
            obstacle_type=OBS_MAP.get(self.combo_obstacle.currentIndex(), "cylinder"),
            obs_D=self.spin_size.value(), obs_angle=float(self.spin_angle.value()))
        self.lbl_area.setText(f"luas ≈ {obstacle_area(cfg):.2f} D²")

    def _set_inputs_enabled(self, enabled):
        for w in (self.combo_obstacle, self.spin_size, self.spin_angle, self.spin_re,
                  self.spin_nx, self.spin_ny, self.radio_fdm, self.radio_fvm,
                  self.radio_cpu, self.radio_gpu, self.radio_fp32, self.radio_fp64):
            w.setEnabled(enabled)
        if not backend.GPU_PRESENT:
            self.radio_gpu.setEnabled(False)    # GPU tetap nonaktif bila tak ada

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI'; font-size: 12px; }
            QGroupBox {
                border: 1px solid #45475a; border-radius: 4px;
                margin-top: 10px; padding: 10px 6px 6px 6px;
                font-weight: bold; color: #89b4fa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QComboBox, QSpinBox {
                background-color: #313244; border: 1px solid #45475a;
                border-radius: 3px; padding: 5px; color: #cdd6f4; font-family: Consolas;
            }
            QComboBox:hover, QSpinBox:hover { border-color: #89b4fa; }
            QComboBox::drop-down { border: none; }
            QRadioButton { spacing: 5px; font-family: Consolas; font-size: 11px; }
            QRadioButton::indicator { width: 14px; height: 14px; }
            QLabel { color: #bac2de; }
            QSplitter::handle { background-color: #45475a; width: 2px; }
            QFrame { background-color: #11111b; border-radius: 4px; }
        """)

    def closeEvent(self, event):
        self._render_timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        if self.export_worker:
            self.export_worker.wait()
        self.plotter.close()
        event.accept()
