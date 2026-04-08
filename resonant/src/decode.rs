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
fn downmix_generic(interleaved: &[f32], ch: usize) -> Vec<f32> {
    let num_frames = interleaved.len() / ch;
    let scale = 1.0 / ch as f32;
    let mut mono = Vec::with_capacity(num_frames);
    for frame in 0..num_frames {
        let start = frame * ch;
        let mut sum = 0.0_f32;
        for c in 0..ch {
            sum += interleaved[start + c];
        }
        mono.push(sum * scale);
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
        let half = _mm_set1_ps(0.5);
        for i in 0..chunks {
            let f_off = i * 8;
            let o_off = i * 4;

            // Load [L0,R0,L1,R1] and [L2,R2,L3,R3]
            let v0 = _mm_loadu_ps(interleaved.as_ptr().add(f_off));
            let v1 = _mm_loadu_ps(interleaved.as_ptr().add(f_off + 4));

            // Deinterleave: lefts = [L0,L1,L2,L3], rights = [R0,R1,R2,R3]
            let lefts = _mm_shuffle_ps::<0b10_00_10_00>(v0, v1);
            let rights = _mm_shuffle_ps::<0b11_01_11_01>(v0, v1);

            let sum = _mm_mul_ps(_mm_add_ps(lefts, rights), half);
            _mm_storeu_ps(mono.as_mut_ptr().add(o_off), sum);
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
        let half = vdupq_n_f32(0.5);
        for i in 0..chunks {
            let f_off = i * 8;
            let o_off = i * 4;

            let pair = vld2q_f32(interleaved.as_ptr().add(f_off));
            let sum = vmulq_f32(vaddq_f32(pair.0, pair.1), half);
            vst1q_f32(mono.as_mut_ptr().add(o_off), sum);
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
}
