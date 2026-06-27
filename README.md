# Simulasi Numerik Vortex Shedding dan Distribusi Panas (2D Navier-Stokes)

Program ini mensimulasikan aliran fluida tak-mampat dua dimensi yang melewati penghalang (obstacle) di dalam saluran menggunakan persamaan **Navier-Stokes** yang dikopel dengan persamaan **transpor energi (suhu)**. Fokus kajian ada pada pola pelepasan vorteks (vortex shedding) dan distribusi panas di belakang penghalang.

Proyek ini menyediakan **dua metode diskritisasi** (Finite Difference Method / FDM dan Finite Volume Method / FVM) yang dapat dipilih dan dibandingkan secara langsung pada kasus yang sama melalui antarmuka GUI Desktop, serta **akselerasi GPU otomatis (CuPy) dengan fallback CPU (Numba)**.

## Backend Komputasi (CPU / GPU)

Backend bisa **dipilih saat runtime** melalui panel "Compute Backend" (radio CPU / GPU) di GUI per simulasi (lihat `src/backend.py`):

| Mode | Engine | Modul |
|:--|:--|:--|
| **CPU** | Numba `@njit(parallel=True)` di seluruh core | `kernels.py` |
| **GPU** | Operasi array vektorisasi CuPy | `gpu_ops.py` |

Medan dialokasikan menggunakan numpy (CPU) atau cupy (GPU); kode solver sama. Bila tidak ada GPU NVIDIA yang terdeteksi, opsi GPU dinonaktifkan secara otomatis dan program memakai CPU.

Catatan: CuPy hanya untuk GPU **NVIDIA** (bukan iGPU Intel/AMD).

**Default dan Override**: Tanpa pengaturan, GPU dipakai bila ada. Dapat dipaksa lewat environment variable:
```bash
FISKOM_BACKEND=cpu  python simulasi.py     # default CPU
FISKOM_BACKEND=gpu  python simulasi.py     # default GPU
```
Atau pilih dari panel GUI.

Kinerja: Pada GPU kelas-masuk (mis. GTX 1650 Ti laptop), solver CPU multicore (Numba) juga bisa sangat cepat karena utilisasi thread penuh. GPU baru akan menunjukkan jarak kecepatan signifikan pada grid resolusi sangat tinggi. Jalankan `python Pengujian/benchmark.py` untuk mengukur metrik di perangkat Anda.

## Struktur Repositori

```text
NavierStokes-2D/
├── README.md                  # Dokumentasi utama (file ini)
├── requirements.txt           # Daftar dependensi Python
├── simulasi.py                # Entry point: membuka aplikasi GUI
├── Pengujian/                 # Skrip validasi fisika dan performa
│   ├── benchmark.py           # Benchmark throughput & FPS (GPU vs CPU)
│   ├── uji_konservasi.py      # Validasi kekekalan massa (divergensi)
│   ├── uji_konvergensi.py      # Uji konvergensi grid
│   ├── uji_stabilitas.py      # Evaluasi kriteria Courant dan Fourier
│   └── validasi_poiseuille.py # Validasi terhadap solusi analitik Poiseuille
├── Utilitas/                  # Skrip alat bantu output data
│   ├── batch_plot.py          # Automasi ekspor plot PNG untuk 4 geometri
│   └── render_mp4.py          # CLI render animasi ke MP4 offline
└── src/                       # Source code utama (modular)
    ├── __init__.py
    ├── backend.py             # Deteksi backend GPU(CuPy)/CPU(Numba)
    ├── config.py              # Parameter simulasi & fisika
    ├── grid.py                # Array medan & geometri obstacle
    ├── kernels.py             # Engine CPU (Numba JIT)
    ├── gpu_ops.py             # Engine GPU (CuPy array shifting)
    ├── solver.py              # Engine utama (Proyeksi Chorin, CD, Nusselt)
    ├── render.py              # Renderer plotting
    └── gui.py                 # Antarmuka Desktop (PyQt6 + PyVista)
```

## Teori Singkat

### Persamaan Pengatur

Aliran fluida inkompresibel diatur oleh persamaan Navier-Stokes:

$$\nabla \cdot \mathbf{u} = 0 \quad \text{(Kontinuitas)}$$

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} \quad \text{(Momentum)}$$

$$\frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla)T = \alpha \nabla^2 T \quad \text{(Energi)}$$

### Bentuk Diskrit (Metode Proyeksi Chorin)

Penyelesaian diskrit (menggunakan Finite Difference orde-2 pada grid kartesian) dibagi menjadi langkah-langkah fraksional (Chorin's Projection):

1. **Langkah Prediksi Momentum (Adveksi-Difusi Kecepatan)**
   Menghitung tebakan kecepatan ($u^*, v^*$). Misal untuk komponen horisontal $u$ di koordinat spasial $(i, j)$:
   $$u_{i,j}^* = u_{i,j}^n + \Delta t \left[ - \left( u_{i,j}^n \frac{u_{i+1,j}^n - u_{i-1,j}^n}{2\Delta x} + v_{i,j}^n \frac{u_{i,j+1}^n - u_{i,j-1}^n}{2\Delta y} \right) + \nu \left( \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{\Delta y^2} \right) \right]$$

2. **Langkah Tekanan (Persamaan Poisson)**
   Mencari medan tekanan ($p^{n+1}$) untuk menyeimbangkan massa fluida agar divergensi nol:
   $$\frac{p_{i+1,j}^{n+1} - 2p_{i,j}^{n+1} + p_{i-1,j}^{n+1}}{\Delta x^2} + \frac{p_{i,j+1}^{n+1} - 2p_{i,j}^{n+1} + p_{i,j-1}^{n+1}}{\Delta y^2} = \frac{\rho}{\Delta t} \left( \frac{u_{i+1,j}^* - u_{i-1,j}^*}{2\Delta x} + \frac{v_{i,j+1}^* - v_{i,j-1}^*}{2\Delta y} \right)$$

3. **Langkah Koreksi (Proyeksi Akhir)**
   Memperbaiki tebakan awal dengan korektor gradien tekanan (contoh untuk $u$):
   $$u_{i,j}^{n+1} = u_{i,j}^* - \frac{\Delta t}{\rho} \left( \frac{p_{i+1,j}^{n+1} - p_{i-1,j}^{n+1}}{2\Delta x} \right)$$

4. **Transpor Energi (Distribusi Suhu)**
   Persamaan energi termal dieksekusi dengan pendekatan diskritisasi spasial adveksi-difusi serupa:
   $$T_{i,j}^{n+1} = T_{i,j}^n + \Delta t \left[ - \left( u_{i,j}^n \frac{T_{i+1,j}^n - T_{i-1,j}^n}{2\Delta x} + v_{i,j}^n \frac{T_{i,j+1}^n - T_{i,j-1}^n}{2\Delta y} \right) + \alpha \left( \frac{T_{i+1,j}^n - 2T_{i,j}^n + T_{i-1,j}^n}{\Delta x^2} + \frac{T_{i,j+1}^n - 2T_{i,j}^n + T_{i,j-1}^n}{\Delta y^2} \right) \right]$$

### Kestabilan Numerik

Karena menggunakan metode eksplisit, komputasi harus secara kaku mematuhi 3 syarat stabilitas mutlak agar tidak meledak (divergen):

1. **Batas Adveksi (Courant-Friedrichs-Lewy / CFL)**
   Fluida tidak boleh melompat lebih dari satu sel grid per langkah waktu:
   $$CFL = U \frac{\Delta t}{\Delta x} \le 1$$

2. **Batas Difusi Viskositas (Fourier Number)**
   Rambatan viskositas tidak boleh mendahului waktu langkah:
   $$Fo = \nu \frac{\Delta t}{\Delta x^2} \le 0.5$$

3. **Batas Difusi Termal (Fourier Suhu)**
   Rambatan panas mengikuti batas serupa agar tidak berosilasi tajam:
   $$Fo_T = \alpha \frac{\Delta t}{\Delta x^2} \le 0.5$$

*(Catatan: Program secara otomatis mencari batas yang paling ketat dan menurunkan nilai langkah waktu $\Delta t$ untuk menjamin komputasi 100% stabil di berbagai input resolusi).*

### Metode Numerik

| Komponen | Pendekatan |
|:---|:---|
| **Grid** | Staggered / MAC (Marker-and-Cell) |
| **Integrasi Waktu** | Euler eksplisit + Metode Proyeksi Chorin |
| **Adveksi (FDM)** | Central Difference (orde-2) |
| **Adveksi (FVM)** | Blended central/upwind (`adv_blend`, deferred-correction) |
| **Difusi** | Laplacian (shared, identik pada grid seragam) |
| **Tekanan** | Solver Poisson iteratif (Jacobi) |
| **Dinding batas** | `freeslip` (default) atau `noslip` |

Vortex shedding membutuhkan porsi Skema Upwind agar tetap stabil namun tidak boleh terlalu difusif. Parameter `adv_blend` (default 0.8) mencampur metode Central (minim difusi) dan sedikit Upwind (stabil) sehingga formasi jalanan vorteks von Karman dapat muncul sempurna.

### Perbandingan FDM vs FVM

| Aspek | FDM | FVM |
|:---|:---|:---|
| Filosofi | Aproksimasi turunan di **titik** | Keseimbangan **fluks** di muka sel |
| Konservasi lokal | Tidak dijamin | Dijamin |
| Adveksi | Central orde-2 | Blended central/upwind (`adv_blend`) |
| Stabilitas | Rawan divergen pada cell-Re tinggi | Stabil (porsi upwind meredam) |

## Persyaratan Sistem

- **Python** versi 3.10 ke atas
- **Hardware**: GPU NVIDIA opsional (akselerasi CuPy). Tanpa GPU akan menggunakan fallback CPU (Numba).
- Pustaka Python:
  ```bash
  pip install -r requirements.txt
  ```
  Untuk akselerasi GPU (NVIDIA), pastikan CuPy terpasang sesuai versi CUDA di sistem Anda:
  ```bash
  pip install cupy-cuda12x      # Untuk CUDA 12.x
  ```

## Cara Menjalankan

### Aplikasi GUI Desktop (Utama)
```bash
python simulasi.py
```
Sebuah jendela desktop akan terbuka. Pilih parameter di panel kiri, lalu tekan "Mulai Simulasi".

### Uji Stabilitas / Konvergensi / Konservasi
Jalankan file di dalam direktori `Pengujian/`:
```bash
python Pengujian/uji_stabilitas.py
```

### Render Animasi MP4 (Offline)
Jalankan utilitas rendering offline:
```bash
python Utilitas/render_mp4.py --field temperature --re 150
```

### Benchmark Performa (Throughput & FPS)
```bash
python Pengujian/benchmark.py
```
Mengukur throughput solver: steps/detik, MLUPS, dan FPS animasi.

## Fitur Proyek

- **Akselerasi GPU Otomatis**: Transisi seamless antara CuPy (NVIDIA) dan CPU (Numba).
- **Beragam Geometri Penghalang**: Silinder, Elips, Persegi, Belah Ketupat, Segitiga, Pelat - dengan ukuran skala dan sudut rotasi yang dapat disesuaikan.
- **Metode Diskritisasi Ganda**: FDM dan FVM, dapat dipilih interaktif dari GUI.
- **Visualisasi Multivariat**: Vortisitas, Suhu, dan Besar Kecepatan.
- **GUI Desktop**: Parameter real-time, grafik langsung PyVista, serta panel diagnostik fisika.
- **Diagnostik Live Fisika**: Drag Coefficient (CD), Lift Coefficient (CL), Strouhal Number, Nusselt Number, divergensi kecepatan, residual Poisson, dan iterasi per detik.

## Referensi

1. Y. A. Cengel dan J. M. Cimbala, Fluid Mechanics: Fundamentals and Applications, 4th ed., McGraw-Hill, 2018.
2. J. H. Ferziger, M. Peric, dan R. L. Street, Computational Methods for Fluid Dynamics, 4th ed., Springer, 2020.
3. J. D. Anderson, Computational Fluid Dynamics: The Basics with Applications, McGraw-Hill, 1995.
4. A. J. Chorin, "Numerical solution of the Navier-Stokes equations", Math. Comput., vol. 22, no. 104, 1968.
5. L. A. Barba dan G. F. Forsyth, "CFD Python: the 12 steps to Navier-Stokes equations", JOSE, vol. 1, no. 9, 2018.
6. M. Schafer dan S. Turek, "Benchmark computations of laminar flow around a cylinder", Flow Simulation with HPC II, 1996.
