# resonant examples

## spectrum_visualiser

Loads an audio file, runs a 4096-point FFT with a Hann window, and prints the 10 strongest frequency peaks.

```bash
cargo run -p resonant-examples --bin spectrum_visualiser -- assets/test.wav
```

Sample output:

```
Loaded: 44100 Hz, 1 ch, 6.68s (294760 samples)

Top 10 frequency peaks:
Rank    Freq (Hz)  Magnitude
------------------------------
1           527.6      51.52
2            64.6      36.53
3           538.3      36.28
...
```

## beat_detector

A simplified onset detector using spectral flux. Computes the STFT (1024-point window, 50% overlap), measures the positive magnitude change between consecutive frames, and flags frames that exceed a threshold of mean + 1.5 standard deviations.

```bash
cargo run -p resonant-examples --bin beat_detector -- assets/test.wav
```

Sample output:

```
Loaded: 44100 Hz, 6.68s

Onset detection (spectral flux > 261.7):
Frame        Time (s)       Flux
----------------------------------
5               0.058      374.0
28              0.325      478.2
...

Detected 33 onsets in 6.68s
```

## spectrogram

Generates a **spectrogram PNG** — a heatmap of frequency vs. time, showing how the spectral content of the audio evolves. Uses a 2048-point window with 75% overlap, clamped to 0–8 kHz for readability.

Outputs `spectrogram.png` in the current directory.

```bash
cargo run -p resonant-examples --bin spectrogram -- assets/test.wav
```

![spectrogram](../spectrogram.png)

## window_comparison

Generates a **window comparison PNG** — overlays the magnitude spectrum of a 1 kHz sine wave under five different window functions (Rectangular, Hann, Hamming, Blackman, Bartlett). Demonstrates the trade-off between main-lobe width and side-lobe suppression.

No audio file needed — the sine wave is synthesised internally. Outputs `window_comparison.png` in the current directory.

```bash
cargo run -p resonant-examples --bin window_comparison
```

![window_comparison](../window_comparison.png)

What to look for:
- **Rectangular** (red): narrowest main lobe, but highest side lobes — most spectral leakage
- **Blackman** (purple): widest main lobe, but side lobes drop fastest — least leakage
- **Hann/Hamming** (blue/green): good middle ground for most applications

## Providing own audio

The file-based examples accept a path as the first argument. Supported formats: WAV, MP3, FLAC, OGG/Vorbis.

```bash
cargo run -p resonant-examples --bin spectrum_visualiser -- path/to/your/file.mp3
```

If no path is given, they default to `assets/test.wav` in the workspace root.
