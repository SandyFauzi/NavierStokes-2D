# Kelompok 17

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CFD: Navier-Stokes](https://img.shields.io/badge/CFD-Navier--Stokes-red.svg)](#)

| Nama | NPM |
|:--|:--|
| Sandy Fauzi A | 140310240054 |
| Anisa Nurhasanah | 140310240001 |
| Choirinnisa Ayu K | 140310240003 |
| Siti Novianti | 140310240002 |

[Bahasa Indonesia](README.md) | [English](README.en.md)

## Demonstrasi Visual

### Karakteristik Vorticity pada 4 Geometri Wajib
![Vorticity Cylinder](results/batch_cylinder_Re140_vorticity.png)
![Vorticity Square](results/batch_square_Re140_vorticity.png)
![Vorticity Triangle](results/batch_triangle_Re140_vorticity.png)
![Vorticity Ellipse](results/batch_ellipse_Re240_vorticity.png)

### Simulasi Video (Evolusi Suhu & Kecepatan)

**1. Cylinder Temperature (Re=500)**
![Cylinder Temperature Re500](results/cylinder_temperature_Re500.gif)

**2. Cylinder Temperature (Re=90)**
![Cylinder Temperature Re90](results/cylinder_temperature_Re90.gif)

## Simulasi Vortex Shedding dan Distribusi Panas 2D

Solver Navier-Stokes dua dimensi untuk aliran incompressible yang melewati sebuah penghalang di dalam saluran. Program menyelesaikan persamaan momentum dan kontinuitas yang dikopel dengan persamaan energi (suhu), lalu menampilkan pola vortex shedding (von Karman vortex street) dan sebaran panas di belakang penghalang.

Simulasi berjalan lewat aplikasi desktop. Pengguna mengatur parameter di panel kiri, menjalankan simulasi, dan melihat medan aliran ditampilkan langsung. Dua metode diskritisasi (FDM dan FVM) serta dua backend komputasi (CPU dan GPU) dapat dipilih dan dibandingkan pada kasus yang sama.

## Fitur

| Fitur | Keterangan |
|:--|:--|
| Dua metode | FDM (central difference) dan FVM (blended central/upwind), dipilih dari GUI |
| Dua backend | CPU (Numba) dan GPU (CuPy), dipilih saat program berjalan |
| Geometri penghalang | silinder, elips, persegi, belah ketupat, segi enam, segitiga, pelat; ukuran dan sudut orientasi dapat diatur |
| Medan tampilan | vorticity, suhu, dan velocity magnitude |
| Colormap | beberapa colormap yang bisa diganti saat simulasi berjalan |
| Tema antarmuka | mode gelap dan terang |
| Metrik fisika | koefisien drag (CD), lift (CL), bilangan Strouhal, bilangan Nusselt, divergensi kecepatan, residual Poisson, langkah per detik |
| Kontrol kecepatan | throttle laju simulasi, dari gerak lambat sampai kecepatan penuh |
| Ekspor | snapshot PNG (lengkap dengan legend parameter) dan animasi MP4 |

## Backend komputasi

Backend dipilih per simulasi lewat panel "Compute Backend" di GUI. Kode solver sama untuk keduanya; yang berbeda hanya pustaka array yang dipakai.

| Mode | Engine | Modul |
|:--|:--|:--|
| CPU | Numba `@njit(parallel=True)`, memakai seluruh core | `src/kernels.py` |
| GPU | CuPy (operasi array dan RawKernel untuk Poisson) | `src/gpu_ops.py` |

CuPy hanya mendukung GPU NVIDIA, bukan iGPU Intel atau AMD. Bila GPU NVIDIA tidak terdeteksi, opsi GPU dinonaktifkan dan program memakai CPU.

Default backend bisa dipaksa lewat environment variable:

```bash
FISKOM_BACKEND=cpu python main.py
FISKOM_BACKEND=gpu python main.py
```

## Catatan performa

GPU tidak selalu lebih cepat dari CPU. Pada grid kecil, seluruh data muat di cache L3 sehingga CPU multicore (Numba) menang. GPU baru unggul pada grid besar, ketika overhead peluncuran kernel sudah teramortisasi dan data tumpah dari cache ke RAM. Pada laptop GTX 1650 Ti yang diuji, titik silang ada di sekitar 800x320 sel: di atas ukuran itu GPU lebih cepat (sampai sekitar 2x pada 1400x560), di bawahnya CPU lebih cepat.

Untuk mengukur di perangkat sendiri:

```bash
python tests/benchmark.py
```

## Struktur proyek

```text
NavierStokes-2D/
├── README.md                  # dokumentasi (Bahasa Indonesia)
├── README.en.md               # dokumentasi (English)
├── requirements.txt           # daftar dependensi Python
├── main.py                # titik masuk: membuka aplikasi GUI
├── src/                       # kode inti program
│   ├── config.py              # parameter simulasi dan fisika
│   ├── backend.py             # deteksi dan pemilihan backend CPU/GPU
│   ├── grid.py                # alokasi medan dan geometri penghalang
│   ├── kernels.py             # engine CPU (Numba JIT)
│   ├── gpu_ops.py             # engine GPU (CuPy)
│   ├── solver.py              # langkah waktu (proyeksi Chorin), CD/CL/Strouhal/Nusselt
│   ├── render.py              # renderer gambar dan video (matplotlib)
│   └── gui.py                 # antarmuka desktop (PyQt6 + PyVista)
├── tests/                 # skrip validasi fisika dan performa
│   ├── benchmark.py           # throughput dan FPS, CPU vs GPU
│   ├── validate_poiseuille.py # uji terhadap solusi analitik Poiseuille
│   ├── test_conservation.py      # kekekalan massa dan positivitas suhu
│   ├── test_convergence.py     # konvergensi grid (orde akurasi)
│   └── test_stability.py      # bilangan Courant dan Fourier
└── utils/                  # alat bantu produksi output
    ├── batch_plot.py          # ekspor PNG otomatis untuk beberapa geometri
    └── render_mp4.py          # render animasi MP4 dari command line
```

Semua output (PNG, MP4) disimpan ke folder `results/` di dalam direktori ini.

## Kebutuhan sistem

- Python 3.10 atau lebih baru
- GPU NVIDIA bersifat opsional; tanpa GPU program memakai CPU

Pasang dependensi:

```bash
pip install -r requirements.txt
```

Untuk akselerasi GPU, pasang CuPy sesuai versi CUDA di sistem Anda:

```bash
pip install cupy-cuda12x      # untuk CUDA 12.x
```

## Cara menjalankan

Aplikasi GUI (utama):

```bash
python main.py
```

Jendela desktop akan terbuka. Atur geometri, Reynolds, resolusi grid, metode, dan backend di panel kiri, lalu tekan tombol Start.

Render animasi MP4 dari command line:

```bash
python utils/render_mp4.py --field temperature --re 150
```

Ekspor sekumpulan gambar PNG untuk beberapa geometri:

```bash
python utils/batch_plot.py
```

Jalankan pengujian (contoh stabilitas):

```bash
python tests/test_stability.py
```

## Teori singkat

### Persamaan pengatur

Aliran incompressible dengan transpor panas pasif diatur oleh:

$$\nabla \cdot \mathbf{u} = 0$$

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$

$$\frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla)T = \alpha \nabla^2 T$$

dengan $\mathbf{u}$ kecepatan, $p$ tekanan, $T$ suhu, $\nu$ viskositas kinematik, dan $\alpha$ difusivitas termal. Bilangan Reynolds didefinisikan $Re = U D / \nu$ dengan $D$ ukuran karakteristik penghalang.

### Metode proyeksi Chorin

Tiap langkah waktu memakai metode fraksional pada grid staggered (MAC):

1. Prediksi kecepatan sementara $\mathbf{u}^{\ast}$ dari suku adveksi dan difusi, mengabaikan tekanan.
2. Selesaikan persamaan Poisson tekanan agar kecepatan akhir bebas divergensi.
3. Koreksi $\mathbf{u}^{\ast}$ dengan gradien tekanan untuk mendapat $\mathbf{u}^{n+1}$.
4. Perbarui medan suhu dengan adveksi dan difusi memakai $\mathbf{u}^{n+1}$.

### Bentuk diskrit lengkap

Tiap turunan didekati dengan beda hingga pada grid seragam (spasi $\Delta x$, $\Delta y$, langkah waktu $\Delta t$). Indeks $i$ mengikuti arah $x$ dan $j$ arah $y$.

Kontinuitas (kecepatan akhir bebas divergensi):

$$\frac{u_{i+1,j} - u_{i-1,j}}{2\Delta x} + \frac{v_{i,j+1} - v_{i,j-1}}{2\Delta y} = 0$$

Momentum, prediksi kecepatan (komponen $u$; komponen $v$ analog):

$$u_{i,j}^{\ast} = u_{i,j}^n + \Delta t \left[ -\left( u \frac{u_{i+1,j} - u_{i-1,j}}{2\Delta x} + v \frac{u_{i,j+1} - u_{i,j-1}}{2\Delta y} \right) + \nu \left( \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2} \right) \right]$$

Poisson tekanan:

$$\frac{p_{i+1,j} - 2p_{i,j} + p_{i-1,j}}{\Delta x^2} + \frac{p_{i,j+1} - 2p_{i,j} + p_{i,j-1}}{\Delta y^2} = \frac{\rho}{\Delta t}\left( \frac{u_{i+1,j}^{\ast} - u_{i-1,j}^{\ast}}{2\Delta x} + \frac{v_{i,j+1}^{\ast} - v_{i,j-1}^{\ast}}{2\Delta y} \right)$$

Koreksi kecepatan:

$$u_{i,j}^{n+1} = u_{i,j}^{\ast} - \frac{\Delta t}{\rho}\frac{p_{i+1,j} - p_{i-1,j}}{2\Delta x}$$

Energi (transpor suhu, memakai $u^{n+1}$ dan $v^{n+1}$):

$$T_{i,j}^{n+1} = T_{i,j}^n + \Delta t \left[ -\left( u \frac{T_{i+1,j} - T_{i-1,j}}{2\Delta x} + v \frac{T_{i,j+1} - T_{i,j-1}}{2\Delta y} \right) + \alpha \left( \frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{\Delta x^2} + \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{\Delta y^2} \right) \right]$$

### Kestabilan numerik

Skema eksplisit harus memenuhi syarat berikut agar tidak divergen. Program menghitung ketiganya, memakai yang paling ketat, dan menurunkan $\Delta t$ secara otomatis.

$$\text{CFL: } \quad U \frac{\Delta t}{\Delta x} \le 1$$

$$\text{Difusi viskos (Fourier): } \quad \nu \frac{\Delta t}{\Delta x^2} \le 0.5$$

$$\text{Difusi termal: } \quad \alpha \frac{\Delta t}{\Delta x^2} \le 0.5$$

### Pilihan metode numerik

| Komponen | Pendekatan |
|:--|:--|
| Grid | staggered / MAC |
| Integrasi waktu | Euler eksplisit dengan proyeksi Chorin |
| Adveksi (FDM) | central difference orde-2 |
| Adveksi (FVM) | campuran central dan upwind (`adv_blend`, deferred correction) |
| Difusi | Laplacian orde-2 |
| Tekanan | solver Poisson iteratif (Jacobi) |
| Dinding | `freeslip` (default) atau `noslip` |

FVM mencampur skema central (difusi numerik rendah) dengan sedikit upwind (stabil). Parameter `adv_blend` (default 0.8) menjaga aliran tetap stabil tanpa terlalu difusif, sehingga von Karman vortex street dapat terbentuk.

### FDM dibanding FVM

| Aspek | FDM | FVM |
|:--|:--|:--|
| Dasar | aproksimasi turunan di titik | keseimbangan fluks di muka sel |
| Konservasi lokal | tidak dijamin | dijamin |
| Adveksi | central orde-2 | campuran central/upwind |
| Stabilitas | rawan divergen pada cell-Reynolds tinggi | lebih stabil karena porsi upwind |

## Pengujian dan validasi

Skrip di folder `tests/`:

- `validate_poiseuille.py` membandingkan profil kecepatan numerik dengan solusi analitik Poiseuille pada saluran.
- `test_conservation.py` memeriksa divergensi kecepatan (kekekalan massa) dan batas suhu (maximum principle).
- `test_convergence.py` mengukur error terhadap resolusi grid untuk menaksir orde akurasi.
- `test_stability.py` melaporkan bilangan Courant dan Fourier serta memantau divergensi sepanjang waktu.
- `benchmark.py` mengukur throughput (langkah/detik, MLUPS) CPU dan GPU pada beberapa ukuran grid.

### Hasil Pengujian (Test Results)

Contoh output pengujian stabilitas numerik dan benchmark CPU vs GPU:

```text
================================================================
  UJI STABILITAS NUMERIK : Navier-Stokes 2D
================================================================
  [Re=150 (default GUI)]  dt = 0.01667
    Courant (CFL adveksi) = 0.2000   [OK, batas < 1]
    Difusi (von Neumann)  = 0.0320   [OK, batas < 0.5]
    Difusi termal         = 0.0451   [OK, batas < 0.5]
    -> 3000 langkah: max|div u| = 2.42e+00, finite = True  ==>  STABIL

================================================================
  UJI KONSERVASI MASSA & POSITIVITAS SUHU
================================================================
Menjalankan simulasi FVM...
  FVM - Max Divergence: 8.50e+00
  FVM - Suhu Terendah : -0.0023 (Seharusnya >= 0.0)
  FVM - Suhu Tertinggi: 1.0000 (Seharusnya <= 1.0)
  [!] FVM MELANGGAR kriteria Positivitas (Maximum Principle).

================================================================
  VALIDASI ANALITIK POISEUILLE
================================================================
  L2 Error Kecepatan terhadap solusi analitik Poiseuille:
  FVM : 1.25e-04 (Lulus Uji)
  FDM : 1.18e-04 (Lulus Uji)

==========================================================================
  BENCHMARK: CPU (Numba) vs GPU (CuPy)
==========================================================================
    200x80       16,000    FVM       606.4      224.0    0.37x
    400x160      64,000    FVM       303.6      265.4    0.87x
    600x240     144,000    FVM       216.5      223.8    1.03x
    800x320     256,000    FVM        95.6      189.6    1.98x
   1000x400     400,000    FVM        60.2      136.6    2.27x
   1400x560     784,000    FVM        35.8       75.8    2.11x
--------------------------------------------------------------------------
  >> GPU mulai MENGUNGGULI CPU di 600x240 (1.03x). Pakai grid >= ini untuk demo GPU>CPU.
==========================================================================
```

## Referensi

1. Y. A. Cengel dan J. M. Cimbala, *Fluid Mechanics: Fundamentals and Applications*, edisi ke-4, McGraw-Hill, 2018.
2. J. H. Ferziger, M. Peric, dan R. L. Street, *Computational Methods for Fluid Dynamics*, edisi ke-4, Springer, 2020.
3. J. D. Anderson, *Computational Fluid Dynamics: The Basics with Applications*, McGraw-Hill, 1995.
4. A. J. Chorin, "Numerical solution of the Navier-Stokes equations", *Math. Comput.*, vol. 22, no. 104, 1968.
5. L. A. Barba dan G. F. Forsyth, "CFD Python: the 12 steps to Navier-Stokes equations", *JOSE*, vol. 1, no. 9, 2018.
6. M. Schafer dan S. Turek, "Benchmark computations of laminar flow around a cylinder", *Flow Simulation with High-Performance Computers II*, 1996.
