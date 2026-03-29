//! Error types for audio analysis operations.

use core::fmt;

/// Errors that can occur during audio analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisError {
    /// The input buffer was empty.
    EmptyInput,
    /// A parameter had an invalid value.
    InvalidParameter {
        /// Which parameter was invalid.
        name: &'static str,
        /// Why it was invalid.
        reason: &'static str,
    },
    /// An FFT operation failed.
    Fft(resonant_fft::FftError),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input is empty"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::Fft(e) => write!(f, "FFT error: {e}"),
        }
    }
}

impl From<resonant_fft::FftError> for AnalysisError {
    fn from(e: resonant_fft::FftError) -> Self {
        Self::Fft(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_empty_input() {
        let e = AnalysisError::EmptyInput;
        assert_eq!(e.to_string(), "input is empty");
    }

    #[test]
    fn display_invalid_parameter() {
        let e = AnalysisError::InvalidParameter {
            name: "window_size",
            reason: "must be positive",
        };
        assert_eq!(
            e.to_string(),
            "invalid parameter `window_size`: must be positive"
        );
    }

    #[test]
    fn display_fft_error() {
        let e = AnalysisError::Fft(resonant_fft::FftError::Empty);
        assert_eq!(e.to_string(), "FFT error: FFT input is empty");
    }

    #[test]
    fn from_fft_error() {
        let fft_err = resonant_fft::FftError::NotPowerOfTwo(3);
        let e: AnalysisError = fft_err.into();
        assert_eq!(
            e,
            AnalysisError::Fft(resonant_fft::FftError::NotPowerOfTwo(3))
        );
    }

    #[test]
    fn clone_and_eq() {
        let e = AnalysisError::EmptyInput;
        assert_eq!(e.clone(), e);
    }
}
