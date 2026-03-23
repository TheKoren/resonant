//! Audio file loading and decoding via symphonia.
//!
//! # Example
//!
//! ```rust,ignore
//! use resonant::AudioFile;
//!
//! let audio = AudioFile::open("assets/test.wav")?;
//! println!("Sample rate: {} Hz", audio.sample_rate());
//! println!("Duration: {:.2}s", audio.duration_secs());
//! println!("Samples (mono): {}", audio.samples_mono().len());
//! ```

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::AudioError;

/// A decoded audio file ready for analysis.
///
/// Stores the full decoded audio as f32 samples. Provides both the original
/// interleaved multichannel data and a mono downmix for analysis APIs.
#[derive(Debug, Clone)]
pub struct AudioFile {
    /// Interleaved samples (all channels).
    samples_interleaved: Vec<f32>,
    /// Mono downmix for analysis.
    samples_mono: Vec<f32>,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of channels.
    channels: u16,
}

impl AudioFile {
    /// Open and decode an audio file.
    ///
    /// Supports WAV, MP3, FLAC, and OGG/Vorbis. The entire file is decoded
    /// into memory as f32 samples.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, AudioError> {
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

        let mono = downmix_to_mono(&all_samples, channels);

        Ok(Self {
            samples_interleaved: all_samples,
            samples_mono: mono,
            sample_rate,
            channels,
        })
    }

    /// Sample rate in Hz (e.g. 44100).
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of audio channels (1 = mono, 2 = stereo).
    #[inline]
    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Duration of the audio in seconds.
    #[inline]
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples_mono.len() as f64 / f64::from(self.sample_rate)
    }

    /// Mono-downmixed samples for analysis. If the source is already mono,
    /// this is the original data.
    #[inline]
    #[must_use]
    pub fn samples_mono(&self) -> &[f32] {
        &self.samples_mono
    }

    /// Raw interleaved samples (all channels).
    #[inline]
    #[must_use]
    pub fn samples_raw(&self) -> &[f32] {
        &self.samples_interleaved
    }

    /// Number of frames (samples per channel).
    #[inline]
    #[must_use]
    pub fn num_frames(&self) -> usize {
        self.samples_mono.len()
    }
}

/// Average all channels down to mono.
fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let ch = channels as usize;
    let num_frames = interleaved.len() / ch;
    let scale = 1.0 / channels as f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_wav_file() {
        let audio = AudioFile::open("../assets/test.wav");
        assert!(audio.is_ok(), "Failed to open test.wav: {:?}", audio.err());
        let audio = audio.ok();
        let audio = audio.as_ref();
        assert_eq!(audio.map(|a| a.sample_rate()), Some(44100));
        assert_eq!(audio.map(|a| a.channels()), Some(1));
        assert!(audio.map_or(false, |a| a.num_frames() > 0));
        assert!(audio.map_or(false, |a| a.duration_secs() > 0.0));
    }

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
    fn open_nonexistent_file() {
        let result = AudioFile::open("nonexistent.wav");
        assert!(result.is_err());
        assert!(
            matches!(result, Err(AudioError::Io(_))),
            "Expected Io error"
        );
    }

    #[test]
    fn samples_not_empty() {
        let audio = AudioFile::open("../assets/test.wav");
        if let Ok(audio) = audio {
            assert!(!audio.samples_mono().is_empty());
            assert!(!audio.samples_raw().is_empty());
            // Mono file: raw == mono
            assert_eq!(audio.samples_raw().len(), audio.samples_mono().len());
        }
    }

    #[test]
    fn duration_reasonable() {
        if let Ok(audio) = AudioFile::open("../assets/test.wav") {
            let dur = audio.duration_secs();
            // test.wav is ~590KB at 44100 Hz mono 16-bit = ~6.7s
            assert!(dur > 1.0, "Duration too short: {dur}");
            assert!(dur < 30.0, "Duration too long: {dur}");
        }
    }
}
