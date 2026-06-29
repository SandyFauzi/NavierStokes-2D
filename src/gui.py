# gui desktop (pyqt6 + pyvista)

import time
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QRadioButton, QButtonGroup, QSplitter, QFrame,
    QScrollArea, QSlider, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

import pyvista as pv
from pyvistaqt import QtInteractor

from .config import SimulationConfig, RESULTS_DIR, results_path
from .solver import NavierStokesSolver
from . import backend

OBSTACLES = ["Cylinder", "Ellipse", "Square", "Diamond", "Hexagon", "Triangle", "Flat Plate"]

# template benchmark grid nx ny
GRID_PRESETS = [
    ("200 x 80   (16k sel, CPU optimal)",   200,  80),
    ("400 x 160  (64k sel)",                400, 160),
    ("600 x 240  (144k sel)",               600, 240),
    ("800 x 320  (256k sel, GPU optimal)",  800, 320),
    ("1000 x 400 (400k sel, GPU optimal)", 1000, 400),
    ("1400 x 560 (784k sel, GPU optimal)", 1400, 560),
]
OBS_MAP = {0: "cylinder", 1: "ellipse", 2: "square", 3: "diamond",
           4: "hexagon", 5: "triangle", 6: "plate"}


class SimulationWorker(QThread):
    # step, time, max|div|, p_resid, sps, CD, CL, St, Nu
    step_done = pyqtSignal(int, float, float, float, float, float, float, float, float)
    finished = pyqtSignal()

    def __init__(self, solver, target_sps=0.0):
        super().__init__()
        self.solver = solver
        self._running = True
        self.target_sps = target_sps        # batas langkah/detik (0 = tak terbatas)

    def run(self):
        cfg = self.solver.cfg
        t_last = time.perf_counter()
        n_last = self.solver.step_count
        sched_t0, sched_n, cur_tgt = time.perf_counter(), 0, self.target_sps
        while self._running and self.solver.step_count < cfg.n_steps:
            self.solver.step()
            # fps limiter
            tgt = self.target_sps
            if tgt != cur_tgt:              # slider berubah -> reset jadwal pacing
                cur_tgt, sched_t0, sched_n = tgt, time.perf_counter(), 0
            if cur_tgt > 0:
                sched_n += 1
                wait = sched_t0 + sched_n / cur_tgt - time.perf_counter()
                if wait > 0:
                    time.sleep(min(wait, 0.05))
            if self.solver.step_count % cfg.plot_every == 0:
                self.solver.update_diagnostics()
                now = time.perf_counter()
                dt = now - t_last
                sps = (self.solver.step_count - n_last) / dt if dt > 0 else 0.0
                t_last, n_last = now, self.solver.step_count
                self.step_done.emit(self.solver.step_count, self.solver.time,
                                    self.solver.div_err,
                                    self.solver.p_resid, sps,
                                    self.solver.current_CD, self.solver.current_CL,
                                    self.solver.current_St, self.solver.current_Nu)
        self.finished.emit()

    def stop(self):
        self._running = False


class ExportWorker(QThread):
    # background render mp4
    progress = pyqtSignal(float)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: SimulationConfig, field: str, out_path: str, n_frames: int, frame_every: int, warmup: int, fps: int, mode: str):
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
        self._dark = True              # tema awal: gelap
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
        self._cmap_override = None       # override colormap dari dropdown (None = auto)
        self._stream_actor = None        # actor streamlines (overlay)
        self._stream_counter = 0

        # pisahkan loop simulasi dan render GUI
        # worker thread simulasi vs timer ui 60 fps
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
        self._apply_theme()           # terapkan tema akhir

    def _build_left_panel(self):
        panel_widget = QWidget()
        panel_widget.setMaximumWidth(340)
        layout = QVBoxLayout(panel_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        hdr = QLabel("NAVIER-STOKES SOLVER")
        hdr.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hdr)

        self.btn_theme = QPushButton()
        self.btn_theme.setToolTip("Ganti tema gelap/terang")
        self.btn_theme.clicked.connect(self._toggle_theme)
        layout.addWidget(self.btn_theme)

        tabs = QTabWidget()
        tab_setup = QWidget()
        tab_vis = QWidget()
        tabs.addTab(tab_setup, "Setup Fisika")
        tabs.addTab(tab_vis, "Kontrol & Visual")
        layout.addWidget(tabs)
        
        l_setup = QVBoxLayout(tab_setup)
        l_setup.setSpacing(8)
        
        l_vis = QVBoxLayout(tab_vis)
        l_vis.setSpacing(8)

        # geometri + orientasi penghalang
        grp1 = QGroupBox("Obstacle Geometry")
        l1 = QVBoxLayout(grp1)
        self.combo_obstacle = QComboBox()
        self.combo_obstacle.addItems(OBSTACLES)
        l1.addWidget(self.combo_obstacle)

        row = QHBoxLayout()
        row.addWidget(QLabel("Size (D)"))
        self.spin_size = QDoubleSpinBox()
        self.spin_size.setRange(0.01, 9999.0)
        self.spin_size.setValue(1.0)
        self.spin_size.setSingleStep(0.25)
        row.addWidget(self.spin_size)
        row.addWidget(QLabel("Angle"))
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-9999, 9999)
        self.spin_angle.setValue(0)
        self.spin_angle.setSingleStep(5)
        self.spin_angle.setSuffix(" °")
        row.addWidget(self.spin_angle)
        l1.addLayout(row)

        self.lbl_area = QLabel("Luas rintangan : 0.00 D2")
        l1.addWidget(self.lbl_area)
        l_setup.addWidget(grp1)

        # update readout luas saat geometri berubah
        for w in (self.combo_obstacle,):
            w.currentIndexChanged.connect(self._update_area)
        for w in (self.spin_size, self.spin_angle):
            w.valueChanged.connect(self._update_area)

        grp2 = QGroupBox("Reynolds Number")
        l2 = QVBoxLayout(grp2)
        self.spin_re = QSpinBox()
        self.spin_re.setRange(1, 9999999)
        self.spin_re.setValue(150)
        self.spin_re.setSingleStep(10)
        l2.addWidget(self.spin_re)
        l_setup.addWidget(grp2)

        grp3 = QGroupBox("Grid Resolution (Template)")
        l3 = QVBoxLayout(grp3)
        # layout grid sizing
        self.spin_nx = QSpinBox(); self.spin_nx.setRange(10, 99999)
        self.spin_ny = QSpinBox(); self.spin_ny.setRange(10, 99999)
        self.combo_grid = QComboBox()
        self.combo_grid.addItems([p[0] for p in GRID_PRESETS])
        l3.addWidget(self.combo_grid)
        self.lbl_grid = QLabel("")
        self.lbl_grid.setStyleSheet("color: #9ca3af;")
        l3.addWidget(self.lbl_grid)
        self.combo_grid.currentIndexChanged.connect(self._on_grid_preset)
        self._on_grid_preset(0)
        l_setup.addWidget(grp3)

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
        l_setup.addWidget(grp4)

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
        l_setup.addWidget(grpb)
        l_setup.addStretch()

        grp5 = QGroupBox("Field Display")
        l5 = QVBoxLayout(grp5)
        self.radio_vort = QRadioButton("Vorticity")
        self.radio_temp = QRadioButton("Temperature")
        self.radio_vel = QRadioButton("Velocity Magnitude")
        self.radio_stream = QRadioButton("Streamlines (garis aliran)")
        self.radio_vort.setChecked(True)
        self.bg_vis = QButtonGroup()
        for r in (self.radio_vort, self.radio_temp, self.radio_vel, self.radio_stream):
            self.bg_vis.addButton(r)
            l5.addWidget(r)

        # pemilih colormap (on-the-fly)
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["Auto", "coolwarm", "viridis", "jet",
                                  "magma", "inferno", "turbo", "RdBu_r"])
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        cmap_row.addWidget(self.combo_cmap)
        l5.addLayout(cmap_row)
        self.lbl_vis = QLabel("")
        self.lbl_vis.setWordWrap(True)
        l5.addWidget(self.lbl_vis)

        l_vis.addWidget(grp5)
        for r in (self.radio_vort, self.radio_temp, self.radio_vel, self.radio_stream):
            r.toggled.connect(self._on_vis_changed)

        grp6 = QGroupBox("Control")
        l6 = QVBoxLayout(grp6)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 40)
        self.slider_speed.setValue(8)
        self.lbl_speed = QLabel("8 langkah/frame")
        self.lbl_speed.setStyleSheet("color: #f38ba8; font-size: 11px;")
        speed_layout.addWidget(self.slider_speed)
        l6.addLayout(speed_layout)
        l6.addWidget(self.lbl_speed, alignment=Qt.AlignmentFlag.AlignRight)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        
        self.btn_start = QPushButton("START SIMULATION")
        self.btn_start.clicked.connect(self._on_start)
        l6.addWidget(self.btn_start)
        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        l6.addWidget(self.btn_pause)
        self.btn_reset = QPushButton("RESET")
        self.btn_reset.clicked.connect(self._on_reset)
        l6.addWidget(self.btn_reset)
        self.btn_export = QPushButton("EXPORT MP4")
        self.btn_export.setToolTip("Render field terpilih ke results/*.mp4 (30 FPS, latar belakang)")
        self.btn_export.clicked.connect(self._on_export)
        l6.addWidget(self.btn_export)
        self.btn_png = QPushButton("EXPORT PNG (frame saat ini)")
        self.btn_png.setToolTip("Snapshot field saat ini ke results/*.png + legend parameter (klik pas vortex muncul)")
        self.btn_png.clicked.connect(self._on_export_png)
        l6.addWidget(self.btn_png)
        self.lbl_export = QLabel("")
        self.lbl_export.setWordWrap(True)
        l6.addWidget(self.lbl_export)
        self.lbl_outdir = QLabel(f"Folder output : {RESULTS_DIR}")
        self.lbl_outdir.setWordWrap(True)
        l6.addWidget(self.lbl_outdir)
        l_vis.addWidget(grp6)

        grp7 = QGroupBox("Diagnostics")
        l7 = QVBoxLayout(grp7)
        mono = QFont("Consolas", 10)
        self.lbl_step = QLabel("step     : 0")
        self.lbl_time = QLabel("time     : 0.0000")
        self.lbl_dt = QLabel("dt       : 0.000000")
        self.lbl_div = QLabel("max|div| : 0.00e+00")
        self.lbl_pres = QLabel("p_resid  : 0.00e+00")
        self.lbl_sps = QLabel("sim sps  : 0.0")
        self.lbl_fps = QLabel("render fps: 0.0")
        self.lbl_gpu = QLabel("backend  : " + backend.backend_label(backend.default_mode()))
        for lbl in (self.lbl_step, self.lbl_time, self.lbl_dt, self.lbl_div,
                    self.lbl_pres, self.lbl_sps, self.lbl_fps, self.lbl_gpu):
            lbl.setFont(mono)
            l7.addWidget(lbl)
        l_vis.addWidget(grp7)
        
        grp8 = QGroupBox("Physics Metrics")
        l8 = QVBoxLayout(grp8)
        self.lbl_cd = QLabel("CD       : 0.0000")
        self.lbl_cl = QLabel("CL       : 0.0000")
        self.lbl_st = QLabel("Strouhal : 0.0000")
        self.lbl_nu = QLabel("Nusselt  : 0.00")
        for lbl in (self.lbl_cd, self.lbl_cl, self.lbl_st, self.lbl_nu):
            lbl.setFont(mono)
            l8.addWidget(lbl)
        l_vis.addWidget(grp8)
        l_vis.addStretch()

        self.spin_nx.valueChanged.connect(self._update_area)
        self.spin_ny.valueChanged.connect(self._update_area)
        self._update_area()

        return panel_widget

    @staticmethod
    def _btn_style(c, ch, text_c="#1e1e2e"):
        return "" # Dihapus agar mengikuti gaya bawaan (Native OS)

    # pyvista

    def _init_pyvista(self):
        self.plotter.set_background(getattr(self, "_plot_bg", "#181825"))
        self.plotter.clear()
        self._grid = self._actor = self._cmap = self._dims = None
        self._stream_actor = None
        self._update_preview()

    def _update_preview(self):
        # pre-render domain fluid dan mask penghalang
        if getattr(self, "plotter", None) is None or self.solver is not None:
            return
        from .grid import _obstacle_mask
        cfg = self._build_config()
        mask = _obstacle_mask(cfg)[1:cfg.ny + 1, 1:cfg.nx + 1]
        field = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
        field[mask] = 1.0
        flat = np.ascontiguousarray(field.ravel(order="C"))
        dark = getattr(self, "_dark", True)
        fluid = "#26324d" if dark else "#e7eef7"     # area fluida (domain)
        solid = "#22d3ee" if dark else "#0b6e99"     # siluet penghalang
        self.plotter.clear()
        grid = pv.ImageData(dimensions=(cfg.nx + 1, cfg.ny + 1, 1),
                            spacing=(cfg.dx, cfg.dy, 1.0))
        grid.cell_data["values"] = flat
        self.plotter.add_mesh(grid, scalars="values", cmap=[fluid, solid],
                              clim=(0, 1), show_scalar_bar=False, lighting=False,
                              interpolate_before_map=False)
        self.plotter.add_text(f"Preview  grid {cfg.nx} x {cfg.ny}   domain {cfg.Lx:g} x {cfg.Ly:g}"
                              "   (tekan START)",
                              position="upper_left", font_size=9,
                              color=getattr(self, "_fg_plot", "white"))
        self._grid = self._actor = self._cmap = self._dims = None
        self.plotter.view_xy()
        self.plotter.render()

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
        if self._cmap_override:
            cmap = self._cmap_override

        data = data.astype(np.float64, copy=True)
        data[self.solver.get_obstacle_mask()] = np.nan      # siluet obstacle
        flat = np.ascontiguousarray(data.ravel(order="C"))

        dims = (cfg.nx + 1, cfg.ny + 1, 1)
        if self._grid is None or self._cmap != cmap or self._dims != dims:
            self._rebuild_mesh(flat, cmap, title, clim, dims, cfg)
        else:
            self._grid.cell_data["values"][:] = flat
            self._actor.mapper.scalar_range = clim
        if getattr(self, "radio_stream", None) is not None and self.radio_stream.isChecked():
            self._stream_counter += 1
            # hitung ulang streamline tiap 30 tick saja biar tidak membebani ui
            if self._stream_actor is None or self._stream_counter % 30 == 0:
                self._update_streamlines()
        self.plotter.render()

        now = time.perf_counter()
        if self._last_render is not None:
            dt = now - self._last_render
            if dt > 0:
                self.lbl_fps.setText(f"render fps: {1.0/dt:6.1f}")
        self._last_render = now

    def _rebuild_mesh(self, flat, cmap, title, clim, dims, cfg):
        self.plotter.clear()
        self._stream_actor = None        # clear() membuang overlay -> reset referensi
        grid = pv.ImageData(dimensions=dims, spacing=(cfg.dx, cfg.dy, 1.0))
        grid.cell_data["values"] = flat
        self._actor = self.plotter.add_mesh(
            grid, scalars="values", cmap=cmap, clim=clim,
            interpolate_before_map=True, lighting=False,
            nan_color="#9399b2", nan_opacity=1.0,
            scalar_bar_args={"title": title, "color": getattr(self, "_fg_plot", "white"),
                             "title_font_size": 13, "label_font_size": 11})
        self._grid = grid
        self._cmap = cmap
        self._dims = dims
        self.plotter.view_xy()

    def _on_vis_changed(self):
        self._grid = None       # cmap/clim berubah -> bangun ulang mesh
        self._update_pyvista()

    def _on_cmap_changed(self, txt):
        self._cmap_override = None if txt == "Auto" else txt
        self._grid = None                       # paksa rebuild dengan cmap baru
        if self.solver is not None:
            self._update_pyvista()

    # streamlines (mode tampilan)

    def _remove_streamlines(self):
        if self._stream_actor is not None:
            try:
                self.plotter.remove_actor(self._stream_actor)
            except Exception:
                pass
            self._stream_actor = None

    def _update_streamlines(self):
        # garis aliran di atas latar kecepatan, kecepatan di-downsample biar ringan
        if self.solver is None:
            return
        self._remove_streamlines()
        try:
            cfg = self.solver.cfg
            ny, nx = cfg.ny, cfg.nx
            u = backend.to_cpu(self.solver.d.u[1:ny+1, 1:nx+1])
            u = 0.5 * (u + backend.to_cpu(self.solver.d.u[1:ny+1, 0:nx]))
            v = backend.to_cpu(self.solver.d.v[1:ny+1, 1:nx+1])
            v = 0.5 * (v + backend.to_cpu(self.solver.d.v[0:ny, 1:nx+1]))
            # grid kasar maks 160 sel arah x, integrasi streamline jadi murah
            stride = max(1, nx // 160)
            u = np.ascontiguousarray(u[::stride, ::stride])
            v = np.ascontiguousarray(v[::stride, ::stride])
            cny, cnx = u.shape
            sx, sy = cfg.dx * stride, cfg.dy * stride
            grid = pv.ImageData(dimensions=(cnx, cny, 1), spacing=(sx, sy, 1.0))
            vec = np.zeros((cnx * cny, 3))
            vec[:, 0] = u.ravel(order="C")
            vec[:, 1] = v.ravel(order="C")
            grid["vectors"] = vec
            lines = grid.streamlines_evenly_spaced_2D(
                vectors="vectors", step_length=0.5,
                separating_distance=2.0, separating_distance_ratio=0.4,
                start_position=(sx, cfg.Ly * 0.5, 0.0))
            self._stream_actor = self.plotter.add_mesh(
                lines, color=getattr(self, "_fg_plot", "white"),
                line_width=1.2, lighting=False)
        except Exception as e:
            self._stream_actor = None
            self.lbl_vis.setText(f"streamlines tdk didukung: {e}")
            self.radio_vel.setChecked(True)         # balik ke mode aman

    # aksi

    def _selected_mode(self):
        return "gpu" if self.radio_gpu.isChecked() else "cpu"
        
    def _on_speed_changed(self, val):
        self.lbl_speed.setText(f"{val} langkah/frame")
        if self.worker is not None:
            self.worker.target_sps = val * 60.0      # live: ubah kecepatan saat jalan

    def _on_start(self):
        if self.solver is None:
            cfg = self._build_config()
            self.solver = NavierStokesSolver(cfg, self._selected_mode())
            self.lbl_dt.setText(f"dt       : {self.solver.dt:.6f}")
            self.lbl_gpu.setText(f"backend  : {backend.backend_label(self.solver.mode)}")
            self._vort_clim = None
            self._grid = None
            self._update_pyvista()
        self.worker = SimulationWorker(self.solver, target_sps=self.slider_speed.value() * 60.0)
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
                         (self.lbl_dt, "dt       : 0.000000"), (self.lbl_div, "max|div| : 0.00e+00"),
                         (self.lbl_pres, "p_resid  : 0.00e+00"), (self.lbl_sps, "sim sps  : 0.0"),
                         (self.lbl_fps, "render fps: 0.0"),
                         (self.lbl_cd, "CD       : 0.0000"), (self.lbl_cl, "CL       : 0.0000"),
                         (self.lbl_st, "Strouhal : 0.0000"), (self.lbl_nu, "Nusselt  : 0.00")]:
            lbl.setText(txt)
        self._init_pyvista()

    def _on_step_done(self, step, t, div_err, p_res, sps, cd, cl, st, nu):
        # hanya update label diagnostik; render ditangani timer 60 FPS
        self.lbl_step.setText(f"step     : {step:,}")
        self.lbl_time.setText(f"time     : {t:.4f}")
        self.lbl_div.setText(f"max|div| : {div_err:.2e}")
        self.lbl_pres.setText(f"p_resid  : {p_res:.2e}")
        self.lbl_sps.setText(f"sim sps  : {sps:7.1f}")
        self.lbl_cd.setText(f"CD       : {cd:+.4f}")
        self.lbl_cl.setText(f"CL       : {cl:+.4f}")
        self.lbl_st.setText(f"Strouhal : {st:.4f}")
        self.lbl_nu.setText(f"Nusselt  : {nu:.2f}")

    def _on_render_tick(self):
        if self.solver is not None:
            self._update_pyvista()

    def _on_finished(self):
        self._render_timer.stop()
        self._update_pyvista()              # gambar frame terakhir
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)

    # export mp4

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
        out = results_path(f"{cfg.obstacle_type}_{field}_Re{int(cfg.Re)}.mp4")
        # n_frames 600 + fps 30 = 20 detik, frame_every 22 biar vorteks bergerak jelas
        self.export_worker = ExportWorker(cfg, field, out, n_frames=600,
                                          frame_every=22, warmup=6000, fps=30,
                                          mode=self._selected_mode())
        self.export_worker.progress.connect(
            lambda f: self.lbl_export.setText(f"export {field}: {f*100:4.0f}%"))
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.start()
        self.btn_export.setEnabled(False)
        self.lbl_export.setText(f"export {field}: 0%")

    def _on_export_done(self, path):
        self.export_worker = None
        self.btn_export.setEnabled(True)
        self.lbl_export.setText(f"tersimpan: {path}")

    def _on_export_failed(self, msg):
        self.export_worker = None
        self.btn_export.setEnabled(True)
        self.lbl_export.setText(f"gagal: {msg}")

    def _on_export_png(self):
        # snapshot field saat ini (frame live) ke PNG bergaya makalah + legend parameter
        if self.solver is None:
            self.lbl_export.setText("solver belum aktif, tekan START dulu")
            return
        try:
            from .render import Renderer
            field = self._selected_field_name()
            cfg = self.solver.cfg
            out = results_path(f"{cfg.obstacle_type}_{field}_Re{int(cfg.Re)}_step{self.solver.step_count}.png")
            self.solver.update_diagnostics()
            r = Renderer(cfg, field)
            r.draw(self.solver)
            r.save_png(out)
            r.close()
            self.lbl_export.setText(f"PNG: {out}")
        except Exception as e:
            self.lbl_export.setText(f"PNG gagal: {e}")

    # utility

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
        self.lbl_area.setText(f"Luas rintangan : {obstacle_area(cfg):.2f} D2")
        self._update_preview()

    def _on_grid_preset(self, idx):
        _, nx, ny = GRID_PRESETS[idx]
        self.spin_nx.setValue(nx)
        self.spin_ny.setValue(ny)
        self.lbl_grid.setText(f"{nx} x {ny} = {nx*ny:,} sel")
        self._update_area()

    def _set_inputs_enabled(self, enabled):
        for w in (self.combo_obstacle, self.spin_size, self.spin_angle, self.spin_re,
                  self.combo_grid, self.radio_fdm, self.radio_fvm,
                  self.radio_cpu, self.radio_gpu, self.radio_fp32, self.radio_fp64):
            w.setEnabled(enabled)
        if not backend.GPU_PRESENT:
            self.radio_gpu.setEnabled(False)    # GPU tetap nonaktif bila tak ada

    def _toggle_theme(self):
        self._dark = not self._dark
        self._apply_theme()

    @staticmethod
    def _theme_qss(dark):
        if dark:
            bg, panel, fg, border, hover, sel, handle = (
                "#1a1a1a", "#262626", "#ededed", "#3a3a3a", "#333333", "#4f4f4f", "#8a8a8a")
        else:
            bg, panel, fg, border, hover, sel, handle = (
                "#f7f7f7", "#ffffff", "#1a1a1a", "#c8c8c8", "#ececec", "#c4c4c4", "#5a5a5a")
        return f"""
        QWidget {{ background: {bg}; color: {fg}; font-family: 'Segoe UI'; font-size: 12px; }}
        QGroupBox {{ background: {panel}; border: 1px solid {border}; border-radius: 6px;
                     margin-top: 9px; padding-top: 6px; font-weight: bold; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QLabel {{ background: transparent; }}
        QPushButton {{ background: {panel}; border: 1px solid {border};
                       border-radius: 5px; padding: 6px; }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:disabled {{ color: {border}; }}
        QComboBox, QSpinBox, QDoubleSpinBox {{ background: {panel};
                       border: 1px solid {border}; border-radius: 4px; padding: 3px; }}
        QComboBox QAbstractItemView {{ background: {panel}; color: {fg};
                       selection-background-color: {sel}; }}
        QRadioButton {{ background: transparent; }}
        QTabWidget::pane {{ border: 1px solid {border}; }}
        QTabBar::tab {{ background: {bg}; padding: 6px 12px; border: 1px solid {border}; }}
        QTabBar::tab:selected {{ background: {panel}; font-weight: bold; }}
        QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: {handle}; width: 14px;
                       margin: -6px 0; border-radius: 7px; }}
        QSplitter::handle {{ background: {border}; }}
        QScrollArea {{ border: none; }}
        """

    def _apply_theme(self):
        dark = getattr(self, "_dark", True)
        self._plot_bg = "#15151c" if dark else "#ffffff"
        self._fg_plot = "white" if dark else "black"
        self.setStyleSheet(self._theme_qss(dark))
        muted = "#9ca3af" if dark else "#5a5a5a"
        if hasattr(self, "lbl_grid"):
            self.lbl_grid.setStyleSheet(f"color: {muted};")
        if hasattr(self, "lbl_speed"):
            self.lbl_speed.setStyleSheet(f"color: {muted}; font-size: 11px;")
        if hasattr(self, "btn_theme"):
            self.btn_theme.setText("Tema: Terang" if dark else "Tema: Gelap")
        # plotter ikut tema (rebuild agar warna teks colorbar berganti)
        if getattr(self, "plotter", None) is not None:
            self.plotter.set_background(self._plot_bg)
            self._grid = None
            if getattr(self, "solver", None) is not None:
                self._update_pyvista()
            else:
                self._init_pyvista()

    def closeEvent(self, event):
        self._render_timer.stop()
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        if self.export_worker:
            self.export_worker.wait()
        self.plotter.close()
        event.accept()
