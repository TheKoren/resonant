//! Error types for the resonant facade.

use std::fmt;

/// Errors returned by the resonant facade.
#[derive(Debug)]
pub enum AudioError {
    /// File could not be opened or read.
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
