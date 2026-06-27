"""Renderer matplotlib bergaya makalah (offscreen) -> PNG & MP4."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SimulationConfig, hasil_path
from .solver import NavierStokesSolver

# gaya per field: getter, colormap, judul, label colorbar, mask obstacle, clim simetris
FIELD_STYLE = {
    "temperature": dict(
        getter="get_temperature", cmap="inferno",
        title="Distribusi Suhu — panas dari penghalang tersebar mengikuti pola vorteks",
        cbar="T (ternormalisasi)", label="penghalang panas",
        mask_obstacle=False, symmetric=False, vmin=0.0),
    "vorticity": dict(
        getter="get_vorticity", cmap="RdBu_r",
        title="Medan Vortisitas ω — Jalanan Vorteks von Kármán",
        cbar="ω  (vortisitas)", label="penghalang",
        mask_obstacle=True, symmetric=True, vmin=None),
    "velocity": dict(
        getter="get_velocity_magnitude", cmap="turbo",
        title="Magnitude Kecepatan |u|",
        cbar="|u| (ternormalisasi)", label="penghalang",
        mask_obstacle=True, symmetric=False, vmin=0.0),
}

_FG = "white"
_BG = "#000000"
_RING = "#22d3ee"


def _clim(data, style):
    if style["symmetric"]:
        m = float(np.percentile(np.abs(data), 99.0)) or 1e-6
        return -m, m
    vmin = style["vmin"] if style["vmin"] is not None else float(np.min(data))
    vmax = float(np.percentile(data, 99.5))
    return vmin, vmax if vmax > vmin else vmin + 1e-6


class Renderer:
    def __init__(self, cfg, field="temperature", figsize=(11, 4.2), dpi=100,
                 annotate=True, interpolation="bilinear"):
        self.cfg = cfg
        self.field = field
        self.style = FIELD_STYLE[field]
        self.annotate = annotate

        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=_BG)
        self.ax = self.fig.add_subplot(111, facecolor=_BG)
        self.ax.set_title(self.style["title"], color=_FG, fontsize=11, fontweight="bold", pad=10)
        self.ax.set_xlabel("x (lattice)", color=_FG, fontsize=10)
        self.ax.set_ylabel("y", color=_FG, fontsize=10)
        self.ax.tick_params(colors=_FG, labelsize=8)
        for s in self.ax.spines.values():
            s.set_color("#555555")

        self._extent = [0, cfg.nx, 0, cfg.ny]
        cmap = plt.get_cmap(self.style["cmap"]).copy()
        cmap.set_bad("#9399b2")        # warna sel obstacle saat di-mask
        self.im = self.ax.imshow(np.zeros((cfg.ny, cfg.nx)), origin="lower",
                                 extent=self._extent, cmap=cmap,
                                 interpolation=interpolation, aspect="equal")
        cb = self.fig.colorbar(self.im, ax=self.ax, fraction=0.025, pad=0.02)
        cb.set_label(self.style["cbar"], color=_FG, fontsize=9)
        cb.ax.yaxis.set_tick_params(color=_FG, labelsize=8)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=_FG)
        cb.outline.set_edgecolor("#555555")

        self._ring = None
        self._annot = None
        self._pbox = None
        self._clim_frozen = None
        self.fig.tight_layout()

    def _param_box(self, solver):
        # legend parameter di pojok kiri-atas (bentuk, Re, Pr, metode, grid, CD, St)
        if self._pbox is not None:
            return
        cfg = self.cfg
        cd_hist = getattr(solver, "history_CD", None)
        if cd_hist and len(cd_hist) > 20:                       # CD rata-rata (paruh kedua)
            cd = float(np.mean(np.asarray(cd_hist)[len(cd_hist) // 2:]))
        else:
            cd = getattr(solver, "current_CD", 0.0) or 0.0
        st = getattr(solver, "current_St", 0.0) or 0.0
        lines = [f"Bentuk : {cfg.obstacle_type}",
                 f"Re = {cfg.Re:g}    Pr = {cfg.Pr:g}",
                 f"metode = {cfg.method.upper()}   grid = {cfg.nx} x {cfg.ny}",
                 f"U = {cfg.U_inf:g}   D = {cfg.obs_D:g}",
                 f"CD_avg = {cd:+.3f}   St = {st:.3f}"]
        ron = getattr(solver, "re_onset", None)
        if ron:
            lines.append(f"Re onset shedding ~ {ron:g}")
        self._pbox = self.ax.text(
            0.012, 0.97, "\n".join(lines), transform=self.ax.transAxes,
            color=_FG, fontsize=8.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#000000", ec=_RING, alpha=0.6))

    def _overlay(self, mask):
        # cincin tepi obstacle + anotasi (sekali)
        if self._ring is None:
            self._ring = self.ax.contour(mask.astype(float), levels=[0.5],
                                         colors=[_RING], linewidths=1.3,
                                         extent=self._extent, origin="lower")
        if self.annotate and self._annot is None:
            cfg = self.cfg
            cx, cy = cfg.obs_cx / cfg.dx, cfg.obs_cy / cfg.dy
            self._annot = self.ax.annotate(
                self.style["label"], xy=(cx, cy),
                xytext=(cx + cfg.nx * 0.10, cy + cfg.ny * 0.32),
                color=_RING, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=_RING, lw=1.2))

    def draw(self, solver, freeze_clim=True):
        # render satu frame -> array RGB uint8
        data = getattr(solver, self.style["getter"])().astype(np.float64)
        mask = solver.get_obstacle_mask()

        if freeze_clim and self._clim_frozen is not None:
            vmin, vmax = self._clim_frozen
        else:
            vmin, vmax = _clim(data, self.style)
            if freeze_clim:
                self._clim_frozen = (vmin, vmax)
        self.im.set_clim(vmin, vmax)

        if self.style["mask_obstacle"]:
            data = data.copy()
            data[mask] = np.nan
        self.im.set_data(data)
        self._overlay(mask)
        self._param_box(solver)

        self.fig.canvas.draw()
        rgb = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        h, w = rgb.shape[:2]
        return rgb[: h - (h % 2), : w - (w % 2)]    # dimensi genap (syarat H.264)

    def save_png(self, path):
        self.fig.savefig(path, facecolor=_BG, dpi=self.fig.dpi, bbox_inches="tight")

    def close(self):
        plt.close(self.fig)


def render_png(cfg, field="temperature", warmup=4000, out_path=None,
               interpolation="bilinear", annotate=True, progress=None, mode=None):
    out_path = out_path or hasil_path(f"snapshot_{field}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    solver = NavierStokesSolver(cfg, mode)
    solver.aero_sample_interval = 1                 # sampel tiap langkah -> St akurat
    sample_aero = cfg.obstacle_type != "none"
    for k in range(warmup):
        solver.step()
        if sample_aero:
            solver.compute_aerodynamics()           # isi riwayat CL utk FFT Strouhal
        if progress and k % 200 == 0:
            progress(k / warmup)
    solver.update_diagnostics()
    r = Renderer(cfg, field, annotate=annotate, interpolation=interpolation)
    r.draw(solver)
    r.save_png(out_path)
    r.close()
    return out_path


def render_mp4(cfg, field="temperature", out_path=None,
               n_frames=300, frame_every=20, warmup=3000, fps=30,
               interpolation="bilinear", annotate=True, progress=None, mode=None):
    # total langkah = warmup + n_frames*frame_every; di-cache lalu diputar pada fps
    import imageio.v2 as imageio
    out_path = out_path or hasil_path(f"animasi_{field}.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    solver = NavierStokesSolver(cfg, mode)

    total = warmup + n_frames * frame_every
    done = 0
    for _ in range(warmup):
        solver.step()
        done += 1
        if progress and done % 200 == 0:
            progress(done / total)

    r = Renderer(cfg, field, annotate=annotate, interpolation=interpolation)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)
    try:
        for _f in range(n_frames):
            for _ in range(frame_every):
                solver.step()
                done += 1
            solver.update_diagnostics()
            writer.append_data(r.draw(solver))
            if progress:
                progress(done / total)
    finally:
        writer.close()
        r.close()
    return out_path
