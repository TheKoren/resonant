//! Key detection via Krumhansl-Schmuckler key profiles.
//!
//! Classifies a sequence of chroma frames as one of 24 musical keys
//! (12 tonics × major/minor) using Pearson correlation against the
//! Krumhansl-Schmuckler (1990) key profiles.
//!
//! Reference: Krumhansl, C. L. (1990). *Cognitive Foundations of Musical Pitch*.
//! Oxford University Press.

use crate::chroma::ChromaVector;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One of the 12 chromatic pitch classes, from C upward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchClass {
    /// C
    C,
    /// C# / D♭
    Cs,
    /// D
    D,
    /// D# / E♭
    Ds,
    /// E
    E,
    /// F
    F,
    /// F# / G♭
    Fs,
    /// G
    G,
    /// G# / A♭
    Gs,
    /// A
    A,
    /// A# / B♭
    As,
    /// B
    B,
}

impl PitchClass {
    /// Short name used in standard notation.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PitchClass::C => "C",
            PitchClass::Cs => "C#",
            PitchClass::D => "D",
            PitchClass::Ds => "D#",
            PitchClass::E => "E",
            PitchClass::F => "F",
            PitchClass::Fs => "F#",
            PitchClass::G => "G",
            PitchClass::Gs => "G#",
            PitchClass::A => "A",
            PitchClass::As => "A#",
            PitchClass::B => "B",
        }
    }

    fn from_index(i: usize) -> Self {
        match i % 12 {
            0 => PitchClass::C,
            1 => PitchClass::Cs,
            2 => PitchClass::D,
            3 => PitchClass::Ds,
            4 => PitchClass::E,
            5 => PitchClass::F,
            6 => PitchClass::Fs,
            7 => PitchClass::G,
            8 => PitchClass::Gs,
            9 => PitchClass::A,
            10 => PitchClass::As,
            _ => PitchClass::B,
        }
    }
}

/// Musical mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Major scale.
    Major,
    /// Natural minor scale.
    Minor,
}

/// Result of a key classification.
#[derive(Debug, Clone)]
pub struct KeyEstimate {
    /// Detected tonic pitch class.
    pub tonic: PitchClass,
    /// Detected mode.
    pub mode: Mode,
    /// Pearson correlation with the best-matching profile (0..1).
    /// Values below ~0.3 indicate low confidence (near-atonal input).
    pub confidence: f32,
}

/// Key detector using Krumhansl-Schmuckler profiles.
///
/// # Examples
///
/// ```
/// use resonant_analysis::key::KeyDetector;
/// use resonant_analysis::chroma::ChromaVector;
///
/// // All-A chroma (strong A, C#, E — A major triad)
/// let mut cv = ChromaVector { bins: [0.0; 12] };
/// cv.bins[9]  = 1.0; // A
/// cv.bins[1]  = 0.8; // C#
/// cv.bins[4]  = 0.6; // E
///
/// let result = KeyDetector::new().detect(&[cv]);
/// assert!(result.is_some());
/// ```
pub struct KeyDetector;

impl KeyDetector {
    /// Creates a new detector.
    #[must_use]
    pub fn new() -> Self {
        KeyDetector
    }

    /// Classifies a sequence of chroma frames and returns the best-matching key.
    ///
    /// Returns `None` if `chroma_frames` is empty.
    #[must_use]
    pub fn detect(&self, chroma_frames: &[ChromaVector]) -> Option<KeyEstimate> {
        if chroma_frames.is_empty() {
            return None;
        }

        // Average frames to a single 12-bin profile.
        let mut profile = [0.0_f32; 12];
        for frame in chroma_frames {
            for (i, &v) in frame.bins.iter().enumerate() {
                profile[i] += v;
            }
        }
        let n = chroma_frames.len() as f32;
        for v in &mut profile {
            *v /= n;
        }

        // Test all 24 key templates, track the best Pearson correlation.
        let mut best_r = f32::NEG_INFINITY;
        let mut best_tonic = 0usize;
        let mut best_mode = Mode::Major;

        for tonic in 0..12 {
            let r_major = pearson(&profile, &rotate(KS_MAJOR, tonic));
            let r_minor = pearson(&profile, &rotate(KS_MINOR, tonic));

            if r_major > best_r {
                best_r = r_major;
                best_tonic = tonic;
                best_mode = Mode::Major;
            }
            if r_minor > best_r {
                best_r = r_minor;
                best_tonic = tonic;
                best_mode = Mode::Minor;
            }
        }

        Some(KeyEstimate {
            tonic: PitchClass::from_index(best_tonic),
            mode: best_mode,
            confidence: best_r.clamp(0.0, 1.0),
        })
    }
}

impl Default for KeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Krumhansl-Schmuckler key profiles (1990)
// ---------------------------------------------------------------------------
//
// Each array is indexed from C (0) to B (11) and represents the expected
// tonal salience of each pitch class relative to a C-rooted key.
// These values are from Table 4.1 of Krumhansl (1990).

const KS_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

const KS_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Rotates `profile` right by `n` positions so that `profile[0]` (root
/// salience) lands at chroma bin `n` (the tonic).
fn rotate(profile: [f32; 12], n: usize) -> [f32; 12] {
    let mut out = [0.0_f32; 12];
    for i in 0..12 {
        out[(i + n) % 12] = profile[i];
    }
    out
}

/// Pearson correlation coefficient between two 12-element arrays.
///
/// Returns a value in [−1, 1]; returns 0.0 if either array has zero variance.
fn pearson(x: &[f32; 12], y: &[f32; 12]) -> f32 {
    let mean_x = x.iter().sum::<f32>() / 12.0;
    let mean_y = y.iter().sum::<f32>() / 12.0;

    let mut num = 0.0_f32;
    let mut ss_x = 0.0_f32;
    let mut ss_y = 0.0_f32;

    for i in 0..12 {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        ss_x += dx * dx;
        ss_y += dy * dy;
    }

    let denom = (ss_x * ss_y).sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        num / denom
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chroma(bins: [f32; 12]) -> ChromaVector {
        ChromaVector { bins }
    }

    // Build a chroma where the notes of a major triad are strong.
    fn major_triad_chroma(root: usize) -> ChromaVector {
        let mut bins = [0.0_f32; 12];
        bins[root % 12] = 1.0; // root
        bins[(root + 4) % 12] = 0.8; // major third
        bins[(root + 7) % 12] = 0.6; // perfect fifth
        ChromaVector { bins }
    }

    fn minor_triad_chroma(root: usize) -> ChromaVector {
        let mut bins = [0.0_f32; 12];
        bins[root % 12] = 1.0; // root
        bins[(root + 3) % 12] = 0.8; // minor third
        bins[(root + 7) % 12] = 0.6; // perfect fifth
        ChromaVector { bins }
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(KeyDetector::new().detect(&[]).is_none());
    }

    #[test]
    fn c_major_triad_detects_c_major() {
        // C=0, E=4, G=7
        let cv = major_triad_chroma(0);
        let est = KeyDetector::new().detect(&[cv]).unwrap();
        assert_eq!(est.tonic, PitchClass::C);
        assert_eq!(est.mode, Mode::Major);
    }

    #[test]
    fn a_major_triad_detects_a_major() {
        // A=9, C#=1, E=4
        let cv = major_triad_chroma(9);
        let est = KeyDetector::new().detect(&[cv]).unwrap();
        assert_eq!(est.tonic, PitchClass::A);
        assert_eq!(est.mode, Mode::Major);
    }

    #[test]
    fn a_major_confidence_above_threshold() {
        let mut bins = [0.0_f32; 12];
        bins[9] = 1.0; // A
        bins[1] = 0.8; // C#
        bins[4] = 0.6; // E
        let est = KeyDetector::new().detect(&[make_chroma(bins)]).unwrap();
        assert!(
            est.confidence > 0.8,
            "A major confidence = {:.3}, expected > 0.8",
            est.confidence
        );
    }

    #[test]
    fn c_minor_triad_detects_c_minor() {
        // C=0, Eb=3, G=7
        let cv = minor_triad_chroma(0);
        let est = KeyDetector::new().detect(&[cv]).unwrap();
        assert_eq!(est.tonic, PitchClass::C);
        assert_eq!(est.mode, Mode::Minor);
    }

    #[test]
    fn atonal_chroma_low_confidence() {
        // All equal bins → no tonal centre
        let cv = make_chroma([1.0; 12]);
        let est = KeyDetector::new().detect(&[cv]).unwrap();
        assert!(
            est.confidence < 0.3,
            "flat chroma confidence = {:.3}, expected < 0.3",
            est.confidence
        );
    }

    #[test]
    fn averages_multiple_frames() {
        // Two frames both suggesting A major
        let cv = major_triad_chroma(9);
        let est = KeyDetector::new().detect(&[cv, cv]).unwrap();
        assert_eq!(est.tonic, PitchClass::A);
        assert_eq!(est.mode, Mode::Major);
    }

    #[test]
    fn pitch_class_names() {
        assert_eq!(PitchClass::C.name(), "C");
        assert_eq!(PitchClass::Cs.name(), "C#");
        assert_eq!(PitchClass::Fs.name(), "F#");
        assert_eq!(PitchClass::B.name(), "B");
    }

    #[test]
    fn rotate_wraps_correctly() {
        // Right-rotate by 1: profile[0]=1.0 moves to out[1], profile[11]=12.0 moves to out[0].
        let profile = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let rotated = rotate(profile, 1);
        assert_eq!(rotated[1], 1.0); // root salience lands at tonic bin 1
        assert_eq!(rotated[0], 12.0); // B salience wraps to position 0
    }

    #[test]
    fn pearson_identical_arrays_is_one() {
        let x = KS_MAJOR;
        assert!((pearson(&x, &x) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pearson_flat_array_is_zero() {
        let flat = [1.0_f32; 12];
        assert_eq!(pearson(&KS_MAJOR, &flat), 0.0);
    }
}
