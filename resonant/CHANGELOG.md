# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `AudioFile::open()` — decode WAV, MP3, FLAC, OGG/Vorbis via symphonia
- `AudioFile::samples_mono()` — mono downmix for analysis
- `AudioFile::samples_raw()` — raw interleaved multichannel access
- `AudioFile::sample_rate()`, `channels()`, `duration_secs()`, `num_frames()`
- `AudioError` enum with `Io`, `UnsupportedFormat`, `Decode`, `NoTrack`, `InvalidParameter`, `Fft` variants
- `FrequencyBin` struct with `frequency_hz`, `magnitude`, `phase`, and `db()` method
- `AudioFile::fft()` — full-signal FFT with Hann window and labelled frequency bins
