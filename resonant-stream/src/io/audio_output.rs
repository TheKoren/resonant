use super::IoError;
use crate::{Chunk, DspNode, StreamError};
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};

/// A [`DspNode`] that sends audio to the default output device.
///
/// `AudioOutput` writes incoming [`Chunk`] samples into a shared buffer that
/// the `cpal` output callback drains to fill the hardware speaker or line-out.
/// After enqueuing the samples it returns an empty chunk, making it a terminal
/// node in a pipeline.
///
/// The underlying `cpal` stream runs on a dedicated background thread because
/// `cpal::Stream` is `!Send` on some platforms (e.g. macOS CoreAudio).
pub struct AudioOutput {
    buffer: Arc<Mutex<VecDeque<f32>>>,
    _keep_alive: SyncSender<()>,
}

impl AudioOutput {
    /// Opens the default audio output device.
    ///
    /// Spawns a background thread that owns the `cpal` stream. Returns once
    /// the stream is confirmed playing, or an [`IoError`] if setup fails.
    pub fn open(sample_rate: u32, channels: u16) -> Result<Self, IoError> {
        let buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
        let thread_buffer = Arc::clone(&buffer);
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel::<Result<(), IoError>>(1);
        let (keep_alive_tx, keep_alive_rx) = std::sync::mpsc::sync_channel::<()>(0);

        std::thread::spawn(move || {
            run_output_stream(sample_rate, channels, thread_buffer, init_tx, keep_alive_rx);
        });

        init_rx
            .recv()
            .map_err(|_| IoError::StreamError("audio output thread panicked during setup"))?
            .map(|()| Self {
                buffer,
                _keep_alive: keep_alive_tx,
            })
    }

    #[cfg(test)]
    pub(crate) fn with_buffer() -> (Self, Arc<Mutex<VecDeque<f32>>>) {
        let (keep_alive_tx, _) = std::sync::mpsc::sync_channel(0);
        let buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
        let node = Self {
            buffer: Arc::clone(&buffer),
            _keep_alive: keep_alive_tx,
        };
        (node, buffer)
    }
}

fn run_output_stream(
    sample_rate: u32,
    channels: u16,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    init_tx: SyncSender<Result<(), IoError>>,
    keep_alive_rx: Receiver<()>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            let _ = init_tx.send(Err(IoError::DeviceNotAvailable));
            return;
        }
    };
    let default_config = match device.default_output_config() {
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
            build_output_stream::<f32>(&device, &stream_config, Arc::clone(&buffer))
        }
        cpal::SampleFormat::I16 => {
            build_output_stream::<i16>(&device, &stream_config, Arc::clone(&buffer))
        }
        cpal::SampleFormat::U16 => {
            build_output_stream::<u16>(&device, &stream_config, Arc::clone(&buffer))
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
            // Block until AudioOutput is dropped (keep_alive_tx drops → recv returns Err).
            let _ = keep_alive_rx.recv();
        }
        Err(_) => {
            let _ = init_tx.send(Err(IoError::StreamError("failed to start output stream")));
        }
    }
}

fn build_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    buffer: Arc<Mutex<VecDeque<f32>>>,
) -> Result<cpal::Stream, IoError>
where
    T: cpal::SizedSample + cpal::FromSample<f32> + Send + 'static,
{
    use cpal::traits::DeviceTrait;
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut guard_opt = buffer.lock().ok();
                for sample in data.iter_mut() {
                    *sample = guard_opt
                        .as_mut()
                        .and_then(|g| g.pop_front())
                        .map_or(T::EQUILIBRIUM, T::from_sample);
                }
            },
            |_err| {},
            None,
        )
        .map_err(|_| IoError::StreamError("failed to build output stream"))
}

impl DspNode for AudioOutput {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        let out_sample_rate = input.sample_rate();
        let out_channels = input.channels();
        let mut guard = self
            .buffer
            .lock()
            .map_err(|_| StreamError::ProcessingError("buffer mutex poisoned".into()))?;
        guard.extend(input.into_data());
        Ok(Chunk::empty(out_sample_rate, out_channels))
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
    fn process_pushes_samples_to_buffer() {
        let (mut node, buffer) = AudioOutput::with_buffer();
        let chunk = Chunk::new(vec![0.1_f32, 0.2, 0.3], 44100, 1);
        let out = node.process(chunk).unwrap_or_else(|e| panic!("{e}"));
        assert!(out.is_empty());
        let guard = buffer.lock().unwrap_or_else(|e| panic!("{e}"));
        let collected: Vec<f32> = guard.iter().copied().collect();
        assert_eq!(collected, vec![0.1_f32, 0.2, 0.3]);
    }

    #[test]
    fn reset_clears_buffer() {
        let (mut node, buffer) = AudioOutput::with_buffer();
        let chunk = Chunk::new(vec![1.0_f32; 512], 44100, 1);
        node.process(chunk).unwrap_or_else(|e| panic!("{e}"));
        node.reset();
        let guard = buffer.lock().unwrap_or_else(|e| panic!("{e}"));
        assert!(guard.is_empty());
    }

    #[test]
    fn returns_empty_chunk_with_input_format() {
        let (mut node, _buffer) = AudioOutput::with_buffer();
        let chunk = Chunk::new(vec![0.0_f32; 4], 48000, 2);
        let out = node.process(chunk).unwrap_or_else(|e| panic!("{e}"));
        assert!(out.is_empty());
        assert_eq!(out.sample_rate(), 48000);
        assert_eq!(out.channels(), 2);
    }

    #[test]
    fn successive_process_calls_accumulate_in_buffer() {
        let (mut node, buffer) = AudioOutput::with_buffer();
        node.process(Chunk::new(vec![1.0_f32], 44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        node.process(Chunk::new(vec![2.0_f32], 44100, 1))
            .unwrap_or_else(|e| panic!("{e}"));
        let guard = buffer.lock().unwrap_or_else(|e| panic!("{e}"));
        let collected: Vec<f32> = guard.iter().copied().collect();
        assert_eq!(collected, vec![1.0_f32, 2.0]);
    }

    #[test]
    #[ignore = "requires audio hardware"]
    fn open_default_device() {
        assert!(AudioOutput::open(44100, 2).is_ok());
    }
}
