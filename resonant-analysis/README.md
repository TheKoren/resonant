# resonant-analysis

High-level audio analysis features for the [resonant](https://crates.io/crates/resonant) workspace.

## Features

| Module | Description |
|--------|-------------|
| `spectral` | Spectral centroid, spread, flatness, rolloff |
| `pitch` | YIN fundamental frequency estimator |
| `onset` | Onset detection via spectral flux + adaptive threshold |
| `tempo` | Tempo estimation via autocorrelation of onset envelope |
| `mfcc` | Mel-frequency cepstral coefficients with deltas |
| `chroma` | 12-bin pitch class profiles (C, C#, ..., B) |

## Quick start

```rust
use resonant_analysis::spectral;

let magnitudes = [0.1, 0.5, 1.0, 0.3];
let frequencies = [0.0, 500.0, 1000.0, 1500.0];

let centroid = spectral::spectral_centroid(&magnitudes, &frequencies).unwrap();
let flatness = spectral::spectral_flatness(&magnitudes).unwrap();
```

### Pitch estimation

```rust
use resonant_analysis::pitch::YinEstimator;

let estimator = YinEstimator::new(44100.0);
let est = estimator.estimate(&samples).unwrap();
if let Some(freq) = est.frequency_hz {
    println!("Detected pitch: {freq:.1} Hz (confidence: {:.0}%)", est.confidence * 100.0);
}
```

### Tempo estimation

```rust
use resonant_analysis::tempo::TempoEstimator;

let estimator = TempoEstimator::new(44100.0);
let est = estimator.estimate(&samples).unwrap();
println!("Tempo: {:.1} BPM (confidence: {:.0}%)", est.bpm, est.confidence * 100.0);
```

### MFCC extraction

```rust
use resonant_analysis::mfcc::MfccExtractor;

let extractor = MfccExtractor::new(44100.0);
let frames = extractor.extract(&samples).unwrap();
let deltas = MfccExtractor::deltas(&frames, 2).unwrap();
```

### Chroma features

```rust
use resonant_analysis::chroma::ChromaExtractor;

let extractor = ChromaExtractor::new(44100.0);
let chroma = extractor.extract(&samples).unwrap();
for cv in &chroma {
    println!("Dominant pitch class: {}", cv.dominant_name());
}
```

## License

MIT OR Apache-2.0
