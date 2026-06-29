import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# render animasi/snapshot ke mp4/png (offline)
#
# contoh:
#   python render_mp4.py --field temperature --obstacle cylinder --re 150
#   python render_mp4.py --field vorticity --obstacle plate --size 1.5 --angle 30
#   python render_mp4.py --png --field temperature --warmup 8000

import argparse
from src.config import SimulationConfig
from src import render, backend

SHAPES = ["cylinder", "ellipse", "square", "diamond", "hexagon", "triangle", "plate"]


def main():
    ap = argparse.ArgumentParser(description="Renderer Navier-Stokes 2D -> MP4/PNG")
    ap.add_argument("--field", default="temperature",
                    choices=["temperature", "vorticity", "velocity"])
    ap.add_argument("--obstacle", default="cylinder", choices=SHAPES)
    ap.add_argument("--size", type=float, default=1.0, help="ukuran penghalang D")
    ap.add_argument("--angle", type=float, default=0.0, help="orientasi penghalang (derajat)")
    ap.add_argument("--re", type=float, default=150.0)
    ap.add_argument("--nx", type=int, default=300)
    ap.add_argument("--ny", type=int, default=120)
    ap.add_argument("--method", default="fvm", choices=["fdm", "fvm"])
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--blend", type=float, default=0.8)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--frame-every", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=6000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--png", action="store_true", help="render satu snapshot PNG")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SimulationConfig(nx=args.nx, ny=args.ny, Re=args.re,
                           obstacle_type=args.obstacle, obs_D=args.size,
                           obs_angle=args.angle, method=args.method, adv_blend=args.blend)

    mode = None if args.backend == "auto" else args.backend
    eff_mode = mode or backend.default_mode()
    print(f"Backend : {backend.backend_label(eff_mode)}")
    print(f"Field   : {args.field} | {args.obstacle} D={args.size} @ {args.angle}° | Re {args.re}")

    def prog(f):
        print(f"\r  progress: {f*100:5.1f}%", end="", flush=True)

    if args.png:
        out = args.out or f"results/{args.obstacle}_{args.field}_Re{int(args.re)}.png"
        path = render.render_png(cfg, field=args.field, warmup=args.warmup,
                                 out_path=out, progress=prog, mode=mode)
    else:
        out = args.out or f"results/{args.obstacle}_{args.field}_Re{int(args.re)}.mp4"
        path = render.render_mp4(cfg, field=args.field, out_path=out,
                                 n_frames=args.frames, frame_every=args.frame_every,
                                 warmup=args.warmup, fps=args.fps, progress=prog, mode=mode)
    print(f"\nTersimpan: {path}")


if __name__ == "__main__":
    main()
