//! Labelled frequency bin output from FFT analysis.

/// A single frequency bin with human-readable labels.
///
/// Produced by [`AudioFile::fft()`](crate::AudioFile::fft). Each bin
/// represents the energy at a specific frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrequencyBin {
    /// Centre frequency of this bin in Hz.
    pub frequency_hz: f32,
    /// Magnitude (linear scale).
    pub magnitude: f32,
    /// Phase angle in radians (−π to π).
    pub phase: f32,
}

impl FrequencyBin {
    /// Magnitude in decibels (dB), referenced to 1.0.
    ///
    /// Returns `f32::NEG_INFINITY` for zero magnitude.
    #[inline]
    #[must_use]
    pub fn db(&self) -> f32 {
        20.0 * self.magnitude.max(f32::MIN_POSITIVE).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_unity() {
        let bin = FrequencyBin {
            frequency_hz: 440.0,
            magnitude: 1.0,
            phase: 0.0,
        };
        assert!((bin.db()).abs() < 1e-4, "1.0 magnitude should be ~0 dB");
    }

    #[test]
    fn db_half() {
        let bin = FrequencyBin {
            frequency_hz: 440.0,
            magnitude: 0.5,
            phase: 0.0,
        };
        // 20*log10(0.5) ≈ -6.02 dB
        assert!((bin.db() - (-6.0206)).abs() < 0.01);
    }

    #[test]
    fn db_zero_magnitude() {
        let bin = FrequencyBin {
            frequency_hz: 440.0,
            magnitude: 0.0,
            phase: 0.0,
        };
        // Should not panic; returns a very negative value
        assert!(bin.db() < -300.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn frequency_bin_serde_roundtrip() {
        let bin = FrequencyBin {
            frequency_hz: 440.0,
            magnitude: 0.5,
            phase: 1.2,
        };
        let json =
            serde_json::to_string(&bin).unwrap_or_else(|e| panic!("serialize FrequencyBin: {e}"));
        let back: FrequencyBin =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize FrequencyBin: {e}"));
        assert_eq!(bin, back);
    }
}
