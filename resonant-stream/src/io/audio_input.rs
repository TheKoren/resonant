use super::IoError;
use crate::{Chunk, DspNode, StreamError};
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};

/// A [`DspNode`] that captures audio from the default input device.
///
/// `AudioInput` ignores any incoming [`Chunk`] and outputs chunks filled with
/// samples from the hardware microphone or line-in. When the internal buffer
/// does not yet hold a full chunk, an empty chunk is returned instead.
///
/// The underlying `cpal` stream runs on a dedicated background thread because
/// `cpal::Stream` is `!Send` on some platforms (e.g. macOS CoreAudio).
pub struct AudioInput {
    sample_rate: u32,
    channels: u16,
    chunk_frames: usize,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    _keep_alive: SyncSender<()>,
}

impl AudioInput {
    /// Opens the default audio input device.
    ///
    /// Spawns a background thread that owns the `cpal` stream. Returns once
    /// the stream is confirmed playing, or an [`IoError`] if setup fails.
    /// Each `process()` call will drain 512 frames from the capture buffer.
    pub fn open(sample_rate: u32, channels: u16) -> Result<Self, IoError> {
        Self::open_with_chunk_frames(sample_rate, channels, 512)
    }

    /// Like [`open`](Self::open) but with an explicit output chunk size.
    ///
    /// `chunk_frames` controls how many audio frames each `process()` call
    /// drains from the internal capture buffer.
    pub fn open_with_chunk_frames(
        sample_rate: u32,
        channels: u16,
        chunk_frames: usize,
    ) -> Result<Self, IoError> {
        let buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
        let buffer_thread = Arc::clone(&buffer);
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel::<Result<(), IoError>>(1);
        let (keep_alive_tx, keep_alive_rx) = std::sync::mpsc::sync_channel::<()>(0);

        std::thread::spawn(move || {
            run_input_stream(sample_rate, channels, buffer_thread, init_tx, keep_alive_rx);
        });

        init_rx
            .recv()
            .map_err(|_| IoError::StreamError("audio input thread panicked during setup"))?
            .map(|()| Self {
                sample_rate,
                channels,
                chunk_frames,
                buffer,
                _keep_alive: keep_alive_tx,
            })
    }

    #[cfg(test)]
    pub(crate) fn with_buffer(
        sample_rate: u32,
        channels: u16,
        chunk_frames: usize,
        samples: Vec<f32>,
    ) -> Self {
        let (keep_alive_tx, _) = std::sync::mpsc::sync_channel(0);
        Self {
            sample_rate,
            channels,
            chunk_frames,
            buffer: Arc::new(Mutex::new(VecDeque::from(samples))),
            _keep_alive: keep_alive_tx,
        }
    }
}

fn run_input_stream(
    sample_rate: u32,
    channels: u16,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    init_tx: SyncSender<Result<(), IoError>>,
    keep_alive_rx: Receiver<()>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            let _ = init_tx.send(Err(IoError::DeviceNotAvailable));
            return;
        }
    };
    let default_config = match device.default_input_config() {
        Ok(c) => c,
        Err(_) => {
            let _ = init_tx.send(Err(IoError::DeviceNotAvailable));
            return;
        }
    };
    let stream_config = cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };
    let stream_result = match default_config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_input_stream::<f32>(&device, &stream_config, Arc::clone(&buffer))
        }
        cpal::SampleFormat::I16 => {
            build_input_stream::<i16>(&device, &stream_config, Arc::clone(&buffer))
        }
        cpal::SampleFormat::U16 => {
            build_input_stream::<u16>(&device, &stream_config, Arc::clone(&buffer))
        }
        _ => Err(IoError::UnsupportedFormat(
            "device sample format not supported",
        )),
    };
    let stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };
    match stream.play() {
        Ok(()) => {
            let _ = init_tx.send(Ok(()));
            // Block until AudioInput is dropped (keep_alive_tx drops → recv returns Err).
            let _ = keep_alive_rx.recv();
        }
        Err(_) => {
            let _ = init_tx.send(Err(IoError::StreamError("failed to start input stream")));
        }
    }
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    buffer: Arc<Mutex<VecDeque<f32>>>,
) -> Result<cpal::Stream, IoError>
where
    T: cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    use cpal::traits::DeviceTrait;
    use cpal::Sample;
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if let Ok(mut guard) = buffer.lock() {
                    for &sample in data {
                        guard.push_back(f32::from_sample(sample));
                    }
                }
            },
            |_err| {},
            None,
        )
        .map_err(|_| IoError::StreamError("failed to build input stream"))
}

impl DspNode for AudioInput {
    fn process(&mut self, _input: Chunk) -> Result<Chunk, StreamError> {
        let needed = self.chunk_frames * self.channels as usize;
        let mut guard = self
            .buffer
            .lock()
            .map_err(|_| StreamError::ProcessingError("buffer mutex poisoned".into()))?;
        if guard.len() < needed {
            return Ok(Chunk::empty(self.sample_rate, self.channels));
        }
        let samples: Vec<f32> = guard.drain(..needed).collect();
        Ok(Chunk::new(samples, self.sample_rate, self.channels))
    }

    fn reset(&mut self) {
        if let Ok(mut guard) = self.buffer.lock() {
            guard.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_empty_when_buffer_insufficient() {
        let mut node = AudioInput::with_buffer(44100, 1, 512, vec![0.0; 256]);
        let out = node
            .process(Chunk::empty(44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(out.is_empty());
    }

    #[test]
    fn drains_exact_chunk_frames() {
        let samples: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let mut node = AudioInput::with_buffer(44100, 1, 512, samples);
        let out = node
            .process(Chunk::empty(44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.frames(), 512);
        assert_eq!(out.data()[0], 0.0);
        assert_eq!(out.data()[511], 511.0);
    }

    #[test]
    fn second_process_drains_next_chunk() {
        let samples: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let mut node = AudioInput::with_buffer(44100, 1, 512, samples);
        let _ = node
            .process(Chunk::empty(44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        let out = node
            .process(Chunk::empty(44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.data()[0], 512.0);
    }

    #[test]
    fn reset_clears_buffer() {
        let mut node = AudioInput::with_buffer(44100, 1, 512, vec![1.0_f32; 1024]);
        node.reset();
        let out = node
            .process(Chunk::empty(44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(out.is_empty());
    }

    #[test]
    fn output_carries_correct_metadata() {
        let samples: Vec<f32> = vec![0.0_f32; 1024];
        let mut node = AudioInput::with_buffer(48000, 2, 256, samples);
        let out = node
            .process(Chunk::empty(48000, 2))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.sample_rate(), 48000);
        assert_eq!(out.channels(), 2);
        assert_eq!(out.frames(), 256);
    }

    #[test]
    #[ignore = "requires audio hardware"]
    fn open_default_device() {
        assert!(AudioInput::open(44100, 1).is_ok());
    }
}
