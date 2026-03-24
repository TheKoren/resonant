use core::fmt;

/// Errors that can occur during stream processing.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamError {
    /// Input chunk has no samples.
    EmptyInput,

    /// Sample rate of the chunk does not match the pipeline's expected rate.
    SampleRateMismatch {
        /// The rate the pipeline expects.
        expected: u32,
        /// The rate the chunk actually carries.
        got: u32,
    },

    /// Channel count of the chunk does not match the pipeline's expected count.
    ChannelMismatch {
        /// The count the pipeline expects.
        expected: u16,
        /// The count the chunk actually carries.
        got: u16,
    },

    /// A node-specific processing error.
    ProcessingError(alloc::string::String),
}

extern crate alloc;

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input chunk is empty"),
            Self::SampleRateMismatch { expected, got } => {
                write!(
                    f,
                    "sample rate mismatch: expected {expected} Hz, got {got} Hz"
                )
            }
            Self::ChannelMismatch { expected, got } => {
                write!(f, "channel count mismatch: expected {expected}, got {got}")
            }
            Self::ProcessingError(msg) => write!(f, "processing error: {msg}"),
        }
    }
}

impl std::error::Error for StreamError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_empty_input() {
        let err = StreamError::EmptyInput;
        assert_eq!(err.to_string(), "input chunk is empty");
    }

    #[test]
    fn display_sample_rate_mismatch() {
        let err = StreamError::SampleRateMismatch {
            expected: 44100,
            got: 48000,
        };
        assert_eq!(
            err.to_string(),
            "sample rate mismatch: expected 44100 Hz, got 48000 Hz"
        );
    }

    #[test]
    fn display_channel_mismatch() {
        let err = StreamError::ChannelMismatch {
            expected: 2,
            got: 1,
        };
        assert_eq!(err.to_string(), "channel count mismatch: expected 2, got 1");
    }

    #[test]
    fn display_processing_error() {
        let err = StreamError::ProcessingError("FFT failed".into());
        assert_eq!(err.to_string(), "processing error: FFT failed");
    }

    #[test]
    fn clone_and_eq() {
        let a = StreamError::EmptyInput;
        let b = a.clone();
        assert_eq!(a, b);
    }
}
