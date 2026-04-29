mod audio_input;
mod audio_output;

pub use audio_input::AudioInput;
pub use audio_output::AudioOutput;

use core::fmt;

/// Errors that can occur when opening or operating hardware audio streams.
#[derive(Debug)]
pub enum IoError {
    /// No default audio device was found on the current host.
    DeviceNotAvailable,
    /// The device does not support the requested sample format.
    UnsupportedFormat(&'static str),
    /// The audio stream could not be started or encountered a fatal error.
    StreamError(&'static str),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotAvailable => write!(f, "no audio device available"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported sample format: {msg}"),
            Self::StreamError(msg) => write!(f, "stream error: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}
