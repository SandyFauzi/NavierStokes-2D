# Simulasi Numerik Vortex Shedding & Distribusi Panas (2D Navier-Stokes)

Program ini mensimulasikan aliran fluida tak-mampat dua dimensi yang melewati penghalang (*obstacle*) di dalam saluran menggunakan persamaan **Navier-Stokes** yang dikopel dengan persamaan **transpor energi (suhu)**. Fokus kajian ada pada pola pelepasan vorteks (*vortex shedding*) dan distribusi panas di belakang penghalang.

Proyek ini menyediakan **dua metode diskritisasi** — **Finite Difference Method (FDM)** dan **Finite Volume Method (FVM)** — yang dapat dipilih dan dibandingkan secara langsung pada kasus yang sama melalui antarmuka GUI Desktop, serta **akselerasi GPU otomatis (CuPy) dengan fallback CPU (Numba)**.

---

## ⚡ Backend Komputasi (CPU / GPU — bisa dipilih)

Backend bisa **dipilih saat runtime** — di GUI ada panel **"Compute Backend"**
(radio CPU / GPU), per simulasi (lihat [`src/backend.py`](src/backend.py)):

| Mode | Engine | Modul |
|:--|:--|:--|
| **CPU** | Numba `@njit(parallel=True)` di seluruh core | `kernels.py` |
| **GPU** | Operasi array CuPy (+ kernel `@cuda.jit` Numba bila toolkit lengkap) | `gpu_ops.py`, `gpu_cuda_kernels.py` |

Medan dialokasikan numpy (CPU) atau cupy (GPU); kode solver sama. **CuPy & Numba
digabung**: CuPy mengelola array GPU + reduksi, Numba jadi engine CPU dan (jika
toolkit CUDA lengkap) kernel `@cuda.jit` di GPU. Bila tak ada GPU NVIDIA, opsi GPU
dinonaktifkan dan program memakai CPU.

> Catatan: CuPy & Numba-CUDA **hanya** untuk GPU **NVIDIA** (bukan iGPU Intel/AMD).

**Default & override**: tanpa pengaturan, GPU dipakai bila ada; paksa lewat env:
```bash
FISKOM_BACKEND=cpu  python simulasi.py     # default CPU
FISKOM_BACKEND=gpu  python simulasi.py     # default GPU
```
Atau pilih dari panel GUI / flag `--backend cpu|gpu|auto` di `render_mp4.py`.
> ⚠️ **Kinerja**: pada GPU kelas-masuk (mis. GTX 1650 Ti laptop), solver banyak
> melakukan peluncuran kernel kecil (loop Poisson), sehingga **CPU multicore
> (Numba) bisa lebih cepat** daripada GPU. GPU baru unggul pada GPU desktop besar
> atau grid sangat tinggi. Jalankan `python benchmark.py` untuk mengukur di
> perangkat Anda, lalu pilih backend lewat `FISKOM_BACKEND`.

---

## 📂 Struktur Repositori

```text
Fiskom PBL/
├── README.md                  # Dokumentasi utama (file ini)
├── .gitignore
├── requirements.txt           # Daftar dependensi Python
├── Materi/                    # Dokumen referensi teori (PDF makalah)
├── Hasil/                     # Output validasi: gambar PNG (auto-generated)
├── src/                       # Source code utama (modular)
│   ├── backend.py             # Deteksi backend GPU(CuPy)/CPU(Numba), ekspos `xp`
│   ├── config.py              # Parameter simulasi & konstanta fisika
│   ├── grid.py                # Array medan (via xp) & geometri obstacle (mask)
│   ├── kernels.py             # Engine CPU — semua kernel @njit (adveksi/difusi/Poisson/BC)
│   ├── gpu_ops.py             # Engine GPU — operasi array CuPy (sama-tanda-tangan)
│   ├── gpu_cuda_kernels.py    # Opsional — kernel @cuda.jit upwind (hybrid CuPy+Numba)
│   ├── solver.py              # Engine utama — Metode Proyeksi Chorin (pilih engine)
│   ├── render.py              # Renderer matplotlib bergaya makalah (PNG & MP4)
│   └── gui.py                 # Antarmuka Desktop (PyQt6 + PyVista) + FPS + Export MP4
├── simulasi.py                # Entry point: membuka aplikasi GUI
├── render_mp4.py              # CLI: render animasi/snapshot ke MP4/PNG (offline)
├── benchmark.py               # Benchmark throughput & FPS (GPU vs CPU)
└── validasi_poiseuille.py     # Validasi solver terhadap solusi analitik
```

---

## 🔬 Teori Singkat

### Persamaan Pengatur

Aliran fluida inkompresibel diatur oleh persamaan Navier-Stokes:

$$\nabla \cdot \mathbf{u} = 0 \quad \text{(Kontinuitas)}$$

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} \quad \text{(Momentum)}$$

$$\frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla)T = \alpha \nabla^2 T \quad \text{(Energi)}$$

### Metode Numerik

| Komponen | Pendekatan |
|:---|:---|
| **Grid** | Staggered / MAC (Marker-and-Cell) |
| **Integrasi Waktu** | Euler eksplisit + Metode Proyeksi Chorin |
| **Adveksi (FDM)** | Central Difference (orde-2) |
| **Adveksi (FVM)** | Blended central/upwind (`adv_blend`, deferred-correction) |
| **Difusi** | Laplacian (shared, identik pada grid seragam) |
| **Tekanan** | Solver Poisson iteratif (Jacobi; di GPU = 1 RawKernel/iterasi) |
| **Dinding atas/bawah** | `wall`: `freeslip` (default, latar bersih) / `noslip` |

> **Vortex shedding**: upwind orde-1 murni terlalu difusif (Reynolds efektif jatuh
> ≪ 47) sehingga wake tetap *steady*. Parameter `adv_blend` (default 0.8) memcampur
> central (minim difusi) + sedikit upwind (stabil) sehingga jalanan vorteks von
> Kármán muncul di grid menengah. Dinding `freeslip` + perturbasi awal kecil
> (`seed_perturbation`) membuat shedding berkembang cepat dengan latar bersih
> seperti gambar makalah.

### Perbandingan FDM vs FVM

| Aspek | FDM | FVM |
|:---|:---|:---|
| Filosofi | Aproksimasi turunan di **titik** | Keseimbangan **fluks** di muka sel |
| Konservasi lokal | ❌ Tidak dijamin | ✅ Dijamin |
| Adveksi | Central orde-2 | Blended central/upwind (`adv_blend`) |
| Stabilitas | Rawan divergen pada cell-Re tinggi | Stabil (porsi upwind meredam) |

---

## ⚙️ Persyaratan Sistem

- **Python** versi 3.10 ke atas
- **Hardware**: GPU NVIDIA **opsional** (akselerasi CuPy). Tanpa GPU → CPU (Numba).
- Pustaka Python:
  ```bash
  pip install -r requirements.txt
  ```
  Untuk akselerasi GPU (NVIDIA), tambahkan CuPy sesuai versi CUDA, mis:
  ```bash
  pip install cupy-cuda12x      # CUDA 12.x
  ```

---

## 🚀 Cara Menjalankan

### Aplikasi GUI Desktop (Utama)
```bash
python simulasi.py
```
Sebuah jendela desktop profesional akan terbuka. Pilih parameter di panel kiri, lalu tekan **"Mulai Simulasi"**.

### Validasi Solver (Aliran Poiseuille)
```bash
python validasi_poiseuille.py
```
Membandingkan profil kecepatan numerik terhadap solusi analitik parabolik.

### Render Animasi / Snapshot (gaya makalah → MP4/PNG)
```bash
python render_mp4.py --field temperature --re 150              # animasi MP4 30 FPS
python render_mp4.py --field vorticity --obstacle plate --angle 30   # pelat miring 30°
python render_mp4.py --png --field temperature                 # snapshot PNG
```
Menghasilkan visual bergaya makalah (latar hitam, colormap, penghalang + label,
colorbar) ke folder `Hasil/`. Karena di-*cache* lalu diputar pada `fps` tetap,
playback **mulus 30/60 FPS** tanpa terganggu kecepatan komputasi. Tombol
**EXPORT MP4** di GUI melakukan hal yang sama untuk field yang sedang ditampilkan.

### Benchmark Performa (Throughput & FPS)
```bash
python benchmark.py
```
Mengukur throughput solver pada grid 200×80: **steps/detik**, **MLUPS**, dan
**FPS** animasi. Bila GPU tersedia, otomatis membandingkan GPU (CuPy) vs CPU
(Numba) beserta *speedup*-nya.

---

## ✨ Fitur

- **Akselerasi GPU Otomatis**: CuPy (NVIDIA) dengan fallback CPU (Numba) transparan
- **7 Geometri Penghalang**: Silinder, Elips, Persegi, Belah Ketupat, Heksagon, Segitiga, Pelat — masing-masing dengan **ukuran/scale** (`obs_D`, dengan readout luas) dan **sudut orientasi** (`obs_angle`)
- **2 Metode Diskritisasi**: FDM dan FVM, dapat dipilih dari GUI
- **3 Mode Visualisasi**: Vortisitas, Suhu, Magnitude Kecepatan + siluet obstacle
- **GUI Desktop**: Panel input parameter + render PyVista *real-time* yang mulus
- **Diagnostik Live**: langkah, waktu, max\|div\|, residual Poisson, **steps/s & render FPS**

---

## 📚 Referensi

1. Y. A. Çengel dan J. M. Cimbala, *Fluid Mechanics: Fundamentals and Applications*, 4th ed., McGraw-Hill, 2018.
2. J. H. Ferziger, M. Perić, dan R. L. Street, *Computational Methods for Fluid Dynamics*, 4th ed., Springer, 2020.
3. J. D. Anderson, *Computational Fluid Dynamics: The Basics with Applications*, McGraw-Hill, 1995.
4. A. J. Chorin, "Numerical solution of the Navier-Stokes equations," *Math. Comput.*, vol. 22, no. 104, 1968.
5. L. A. Barba dan G. F. Forsyth, "CFD Python: the 12 steps to Navier-Stokes equations," *JOSE*, vol. 1, no. 9, 2018.
6. M. Schäfer dan S. Turek, "Benchmark computations of laminar flow around a cylinder," *Flow Simulation with HPC II*, 1996.
