//! Audio file decoding and channel downmix.
//!
//! Wraps symphonia for file I/O and provides SIMD-accelerated mono downmix.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::AudioError;

/// Decode an audio file from disk into interleaved f32 samples.
///
/// Returns `(interleaved_samples, sample_rate, channels)`.
pub(crate) fn decode_path<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32, u16), AudioError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    let track = format.default_track().ok_or(AudioError::NoTrack)?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| AudioError::Decode("missing sample rate".to_string()))?;

    let channels = codec_params
        .channels
        .map(|ch| ch.count() as u16)
        .unwrap_or(1);

    let mut decoder =
        symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break; // end of stream
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let audio_buf = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };

        if sample_buf.is_none() {
            let spec = *audio_buf.spec();
            let duration = audio_buf.capacity() as u64;
            sample_buf = Some(SampleBuffer::new(duration, spec));
        }

        if let Some(buf) = &mut sample_buf {
            buf.copy_interleaved_ref(audio_buf);
            all_samples.extend_from_slice(buf.samples());
        }
    }

    Ok((all_samples, sample_rate, channels))
}

/// Downmix interleaved samples to mono using ITU-R BS.1770-4 channel weights.
///
/// For mono, returns a copy of the input unchanged.
/// For stereo, sums L and R then scales by `1 / sqrt(2)` to preserve the
/// mean-square level that BS.1770-4 assigns to a two-channel signal
/// (`output[i] = (L[i] + R[i]) / sqrt(2)`).
/// For >2 channels, falls back to an equal-weight average; accurate per-channel
/// weighting (LFE exclusion, Ls/Rs +3 dB) requires knowing the channel layout,
/// which is not available from the channel count alone.
pub(crate) fn downmix_bs1770(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    if channels == 2 {
        let num_frames = interleaved.len() / 2;
        // SAFETY: 1.0 / sqrt(2) is a compile-time constant; no UB.
        let scale = 1.0_f32 / 2.0_f32.sqrt();
        let mut out = Vec::with_capacity(num_frames);
        for frame in 0..num_frames {
            out.push((interleaved[frame * 2] + interleaved[frame * 2 + 1]) * scale);
        }
        return out;
    }
    downmix_to_mono(interleaved, channels)
}

/// Average all channels down to mono.
///
/// The stereo (2-channel) case uses a SIMD-accelerated path on x86_64 and
/// aarch64. Arbitrary channel counts fall back to scalar summation.
pub(crate) fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    if channels == 2 {
        return downmix_stereo(interleaved);
    }
    downmix_generic(interleaved, channels as usize)
}

/// Scalar N-channel downmix.
fn downmix_generic(interleaved: &[f32], num_channels: usize) -> Vec<f32> {
    let num_frames = interleaved.len() / num_channels;
    let scale = 1.0 / num_channels as f32;
    let mut mono = Vec::with_capacity(num_frames);
    for frame in 0..num_frames {
        let start = frame * num_channels;
        let mut channel_sum = 0.0_f32;
        for channel_idx in 0..num_channels {
            channel_sum += interleaved[start + channel_idx];
        }
        mono.push(channel_sum * scale);
    }
    mono
}

/// SIMD-accelerated stereo-to-mono downmix.
fn downmix_stereo(interleaved: &[f32]) -> Vec<f32> {
    let num_frames = interleaved.len() / 2;
    let mut mono = vec![0.0_f32; num_frames];
    dispatch_stereo_downmix(interleaved, &mut mono);
    mono
}

#[cfg(target_arch = "x86_64")]
fn dispatch_stereo_downmix(interleaved: &[f32], mono: &mut [f32]) {
    downmix_stereo_x86(interleaved, mono);
}

#[cfg(target_arch = "aarch64")]
fn dispatch_stereo_downmix(interleaved: &[f32], mono: &mut [f32]) {
    downmix_stereo_neon(interleaved, mono);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dispatch_stereo_downmix(interleaved: &[f32], mono: &mut [f32]) {
    downmix_stereo_scalar(interleaved, mono);
}

#[allow(dead_code)] // fallback for non-x86/non-aarch64
fn downmix_stereo_scalar(interleaved: &[f32], mono: &mut [f32]) {
    for (i, o) in mono.iter_mut().enumerate() {
        *o = (interleaved[i * 2] + interleaved[i * 2 + 1]) * 0.5;
    }
}

#[cfg(target_arch = "x86_64")]
fn downmix_stereo_x86(interleaved: &[f32], mono: &mut [f32]) {
    use std::arch::x86_64::*;

    let num_frames = mono.len();
    let chunks = num_frames / 4;
    let remainder = num_frames % 4;

    // SAFETY: SSE2 is guaranteed on x86_64. We load 8 f32s (4 stereo frames)
    // per iteration and produce 4 mono samples.
    unsafe {
        let half_scale = _mm_set1_ps(0.5);
        for i in 0..chunks {
            let frame_offset = i * 8;
            let output_offset = i * 4;

            // Load [L0,R0,L1,R1] and [L2,R2,L3,R3]
            let v0 = _mm_loadu_ps(interleaved.as_ptr().add(frame_offset));
            let v1 = _mm_loadu_ps(interleaved.as_ptr().add(frame_offset + 4));

            // Deinterleave: lefts = [L0,L1,L2,L3], rights = [R0,R1,R2,R3]
            let lefts = _mm_shuffle_ps::<0b10_00_10_00>(v0, v1);
            let rights = _mm_shuffle_ps::<0b11_01_11_01>(v0, v1);

            let mixed_frames = _mm_mul_ps(_mm_add_ps(lefts, rights), half_scale);
            _mm_storeu_ps(mono.as_mut_ptr().add(output_offset), mixed_frames);
        }
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = (tail_start + i) * 2;
        mono[tail_start + i] = (interleaved[idx] + interleaved[idx + 1]) * 0.5;
    }
}

#[cfg(target_arch = "aarch64")]
fn downmix_stereo_neon(interleaved: &[f32], mono: &mut [f32]) {
    use std::arch::aarch64::*;

    let num_frames = mono.len();
    let chunks = num_frames / 4;
    let remainder = num_frames % 4;

    // SAFETY: NEON is guaranteed on aarch64. vld2q_f32 deinterleaves 8 f32s
    // into two 4-wide vectors (lefts and rights).
    unsafe {
        let half_scale = vdupq_n_f32(0.5);
        for i in 0..chunks {
            let frame_offset = i * 8;
            let output_offset = i * 4;

            let pair = vld2q_f32(interleaved.as_ptr().add(frame_offset));
            let mixed_frames = vmulq_f32(vaddq_f32(pair.0, pair.1), half_scale);
            vst1q_f32(mono.as_mut_ptr().add(output_offset), mixed_frames);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = (tail_start + i) * 2;
        mono[tail_start + i] = (interleaved[idx] + interleaved[idx + 1]) * 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_downmix_passthrough() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let mono = downmix_to_mono(&samples, 1);
        assert_eq!(mono, samples);
    }

    #[test]
    fn mono_downmix_stereo() {
        // Stereo: L=1.0 R=0.0, L=0.0 R=1.0
        let interleaved = vec![1.0, 0.0, 0.0, 1.0];
        let mono = downmix_to_mono(&interleaved, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bs1770_mono_passthrough() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let out = downmix_bs1770(&samples, 1);
        assert_eq!(out, samples);
    }

    #[test]
    fn bs1770_stereo_identical_channels() {
        // L = R = A → output = A * sqrt(2)
        let a = 0.5_f32;
        let interleaved = vec![a, a, a, a]; // two frames of [L=A, R=A]
        let out = downmix_bs1770(&interleaved, 2);
        let expected = a * 2.0_f32.sqrt();
        assert_eq!(out.len(), 2);
        for &s in &out {
            assert!((s - expected).abs() < 1e-6, "got {s}, expected {expected}");
        }
    }

    #[test]
    fn bs1770_stereo_right_silent() {
        // L = A, R = 0 → output = A / sqrt(2)
        let a = 0.5_f32;
        let interleaved = vec![a, 0.0, a, 0.0];
        let out = downmix_bs1770(&interleaved, 2);
        let expected = a / 2.0_f32.sqrt();
        assert_eq!(out.len(), 2);
        for &s in &out {
            assert!((s - expected).abs() < 1e-6, "got {s}, expected {expected}");
        }
    }

    #[test]
    fn bs1770_stereo_full_vs_half_mean_square_ratio() {
        // Full stereo (L=R=A): mean-sq = 2*A^2
        // L-only   (L=A,R=0): mean-sq = A^2/2
        // Ratio = 4 → 6 dB, reflecting both channels contributing
        let a = 0.5_f32;
        let full = downmix_bs1770(&vec![a, a, a, a], 2);
        let half = downmix_bs1770(&vec![a, 0.0, a, 0.0], 2);

        let ms_full: f32 = full.iter().map(|x| x * x).sum::<f32>() / full.len() as f32;
        let ms_half: f32 = half.iter().map(|x| x * x).sum::<f32>() / half.len() as f32;
        let ratio_db = 10.0 * (ms_full / ms_half).log10();
        assert!(
            (ratio_db - 6.0).abs() < 0.1,
            "expected 6 dB ratio, got {ratio_db:.3} dB"
        );
    }
}
