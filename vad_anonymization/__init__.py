"""vad_anonymization

Reusable building blocks for the human-speech anonymization task
(Task 1 of the dog-sound-recording project): loading and resampling
audio, running Silero VAD to detect speech, and constructing synthetic
speech/noise mixtures with known ground truth for testing.

Modules:
    audio_io   - loading, resampling, and saving audio as tensors.
    detection  - a thin, reusable wrapper around Silero VAD.
    mixing     - building synthetic overlap test cases with known timing.
"""

__all__ = ["audio_io", "detection", "mixing"]
