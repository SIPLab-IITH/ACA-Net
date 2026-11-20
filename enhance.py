#!/usr/bin/env python3
"""Speech enhancement with a pre-trained ACA-Net model.

Enhance a single file:

    python enhance.py --model pre_trained_checkpoints/non-causal/Valentini/acanet_m.pt noisy.wav enhanced.wav

Enhance every .wav in a directory:

    python enhance.py --model pre_trained_checkpoints/non-causal/Valentini/acanet_m.pt --in-dir noisy/ --out-dir enhanced/

The models operate on 16 kHz mono audio. The same command works for the causal
and the non-causal models; the model file is fully self-contained.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000


def enhance_file(model, device, src, dst):
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{src}: expected {SAMPLE_RATE} Hz audio, got {sr} Hz; "
                         f"please resample to {SAMPLE_RATE} Hz first.")
    noisy = audio.mean(axis=1)  # mix down to mono
    with torch.no_grad():
        x = torch.from_numpy(noisy).unsqueeze(0).to(device)
        enhanced = model(x)[0].cpu().numpy()
    peak = float(np.abs(enhanced).max())
    if peak > 1.0:  # scale down only to avoid clipping on write
        enhanced = enhanced / peak
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), enhanced, SAMPLE_RATE, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to a model file, e.g. models/acanet_m.pt")
    parser.add_argument("input", nargs="?", help="noisy input .wav")
    parser.add_argument("output", nargs="?", help="enhanced output .wav")
    parser.add_argument("--in-dir", help="enhance every .wav in this directory")
    parser.add_argument("--out-dir", help="directory for the enhanced files (with --in-dir)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = torch.jit.load(args.model, map_location=device).eval()

    if args.in_dir:
        if not args.out_dir:
            parser.error("--out-dir is required with --in-dir")
        in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
        files = sorted(in_dir.glob("*.wav"))
        if not files:
            raise SystemExit(f"no .wav files found in {in_dir}")
        for f in files:
            enhance_file(model, device, f, out_dir / f.name)
        print(f"enhanced {len(files)} file(s) -> {out_dir}")
    else:
        if not (args.input and args.output):
            parser.error("provide an input and an output .wav, or use --in-dir/--out-dir")
        enhance_file(model, device, args.input, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
