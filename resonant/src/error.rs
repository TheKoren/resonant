//! Error types for the resonant facade.

use std::fmt;

#[cfg(feature = "serde")]
mod io_error_serde {
    use serde::Serializer;
    pub fn serialize<S: Serializer>(e: &std::io::Error, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&e.to_string())
    }
}

/// Errors returned by the resonant facade.
///
/// Implements [`serde::Serialize`] (but not `Deserialize`) when the `serde`
/// feature is enabled. The `Io` variant serialises as the error's `Display` string.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum AudioError {
    /// File could not be opened or read.
    #[cfg_attr(feature = "serde", serde(serialize_with = "io_error_serde::serialize"))]
    Io(std::io::Error),
    /// The audio format or codec is not supported.
    UnsupportedFormat(String),
    /// The decoder encountered corrupt or malformed data.
    Decode(String),
    /// No audio tracks found in the file.
    NoTrack,
    /// An invalid parameter was provided (e.g. zero window size).
    InvalidParameter(String),
    /// FFT operation failed.
    Fft(resonant_fft::FftError),
    /// An analysis operation failed.
    Analysis(resonant_analysis::AnalysisError),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::Decode(msg) => write!(f, "decode error: {msg}"),
            Self::NoTrack => write!(f, "no audio track found in file"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Fft(e) => write!(f, "FFT error: {e}"),
            Self::Analysis(e) => write!(f, "analysis error: {e}"),
        }
    }
}

impl std::error::Error for AudioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Fft(_) => None,
            _ => None,
        }
    }
}

impl From<std::io::Error> for AudioError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<resonant_fft::FftError> for AudioError {
    fn from(e: resonant_fft::FftError) -> Self {
        Self::Fft(e)
    }
}

impl From<resonant_analysis::AnalysisError> for AudioError {
    fn from(e: resonant_analysis::AnalysisError) -> Self {
        Self::Analysis(e)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    use super::*;

    #[cfg(feature = "serde")]
    #[test]
    fn audio_error_serialize_to_json() {
        let cases: &[AudioError] = &[
            AudioError::NoTrack,
            AudioError::UnsupportedFormat("mp4".to_string()),
            AudioError::Decode("corrupt frame".to_string()),
            AudioError::InvalidParameter("window size must be power of two".to_string()),
            AudioError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file.wav",
            )),
        ];
        for e in cases {
            let json = serde_json::to_string(e)
                .unwrap_or_else(|err| panic!("serialize AudioError: {err}"));
            assert!(!json.is_empty());
        }
    }
}

impl From<symphonia::core::errors::Error> for AudioError {
    fn from(e: symphonia::core::errors::Error) -> Self {
        use symphonia::core::errors::Error as SE;
        match e {
            SE::IoError(io) => Self::Io(io),
            SE::Unsupported(msg) => Self::UnsupportedFormat(msg.to_string()),
            SE::DecodeError(msg) => Self::Decode(msg.to_string()),
            SE::LimitError(msg) => Self::Decode(format!("limit exceeded: {msg}")),
            SE::ResetRequired => Self::Decode("decoder reset required".to_string()),
            SE::SeekError(_) => Self::Decode("seek error".to_string()),
        }
    }
}
