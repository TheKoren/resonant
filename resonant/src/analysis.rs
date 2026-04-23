//! High-level analysis result returned by [`AudioFile::analyse()`](crate::AudioFile::analyse).

use resonant_analysis::key::KeyEstimate;
use resonant_analysis::mfcc::MfccFrame;

/// Summary returned by [`AudioFile::analyse()`](crate::AudioFile::analyse).
///
/// All fields are computed in a single pass over the mono signal.
///
/// # Examples
///
/// ```rust,ignore
/// use resonant::AudioFile;
///
/// let result = AudioFile::open("track.wav")?.analyse()?;
/// if let Some(bpm) = result.bpm {
///     println!("Tempo: {bpm:.1} BPM (confidence: {:.2})", result.bpm_confidence);
/// }
/// if let Some(key) = &result.key {
///     println!("Key: {} {}", key.tonic.name(), if key.mode == resonant::Mode::Major { "major" } else { "minor" });
/// }
/// println!("Loudness: {:?} LUFS", result.loudness_lufs);
/// println!("Peak: {:.1} dBFS  RMS: {:.1} dBFS", result.peak_db, result.rms_db);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalysisResult {
    /// Estimated tempo in BPM. `None` if confidence is below threshold.
    pub bpm: Option<f32>,
    /// Tempo estimator confidence in `[0, 1]`.
    pub bpm_confidence: f32,
    /// Detected musical key. `None` if chroma confidence is below 0.3.
    pub key: Option<KeyEstimate>,
    /// Onset timestamps in seconds.
    pub onsets: Vec<f32>,
    /// Integrated loudness in LUFS per ITU-R BS.1770-4.
    ///
    /// `None` if the signal is too short for gating, or the sample rate is not
    /// supported by the K-weighting filter (supported: 44100, 48000, 88200, 96000 Hz).
    pub loudness_lufs: Option<f32>,
    /// Peak level in dBFS. −120.0 for silence.
    pub peak_db: f32,
    /// RMS level in dBFS. −120.0 for silence.
    pub rms_db: f32,
    /// MFCC frames (one per STFT hop). Empty if the signal is too short.
    pub mfcc: Vec<MfccFrame>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use resonant_analysis::key::{Mode, PitchClass};

    #[test]
    fn analysis_result_fields_accessible() {
        let r = AnalysisResult {
            bpm: Some(120.0),
            bpm_confidence: 0.85,
            key: Some(KeyEstimate {
                tonic: PitchClass::A,
                mode: Mode::Major,
                confidence: 0.9,
            }),
            onsets: vec![0.1, 0.6, 1.1],
            loudness_lufs: Some(-14.0),
            peak_db: -0.1,
            rms_db: -18.0,
            mfcc: vec![],
        };
        assert_eq!(r.bpm, Some(120.0));
        assert_eq!(r.onsets.len(), 3);
        assert!(r.key.is_some());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn analysis_result_serde_roundtrip() {
        let r = AnalysisResult {
            bpm: Some(120.0),
            bpm_confidence: 0.85,
            key: None,
            onsets: vec![0.1, 0.5],
            loudness_lufs: Some(-14.0),
            peak_db: -0.1,
            rms_db: -18.0,
            mfcc: vec![],
        };
        let json =
            serde_json::to_string(&r).unwrap_or_else(|e| panic!("serialize AnalysisResult: {e}"));
        let back: AnalysisResult = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize AnalysisResult: {e}"));
        assert_eq!(back.bpm, r.bpm);
        assert_eq!(back.onsets, r.onsets);
        assert!(back.key.is_none());
    }
}
