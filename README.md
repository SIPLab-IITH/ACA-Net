# ACA-Net

Official code for **ACA-Net**: **Significance of Local and Global Context for Low-Complexity Speech Enhancement**

Supplementary material: <https://siplab-iith.github.io/ACA-Net/>

This release provides the trained models and a script to enhance noisy
speech. The full training and evaluation scripts will be released upon publication of our work.

## Models

Five configurations are provided — XT, T, S, M, and L (increasing in size and
quality) — in causal and non-causal versions, trained on VoiceBank+DEMAND and on
DNS-2020:

```
pre_trained_checkpoints/
├── non-causal/
│   ├── Valentini/   acanet_xt.pt  acanet_t.pt  acanet_s.pt  acanet_m.pt  acanet_l.pt
│   └── DNS/         acanet_m.pt   acanet_l.pt
└── causal/
    └── Valentini/   acanet_xt.pt  acanet_t.pt  acanet_s.pt  acanet_m.pt  acanet_l.pt
```

The causal models use only past and current frames and have an algorithmic
latency of 25 ms (the STFT window).

## Installation

```
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.0+.

## Usage

Enhance a single file:

```
python enhance.py --model pre_trained_checkpoints/non-causal/Valentini/acanet_m.pt noisy.wav enhanced.wav
```

Enhance every `.wav` in a directory:

```
python enhance.py --model pre_trained_checkpoints/non-causal/Valentini/acanet_m.pt --in-dir noisy/ --out-dir enhanced/
```

Run on a GPU with `--device cuda`. The same command works for every model; pick a
larger model for higher quality or a smaller one for lower cost.

## Notes

- Input audio must be 16 kHz. Resample first if needed.
- Multi-channel input is mixed down to mono.
- The output is written as 16-bit PCM at 16 kHz.

