import numpy as np
import pytest

from codec.current.wrapper import CODEBOOK_SIZE, NUM_CODEBOOKS, SAMPLE_RATE, MimiCodec


@pytest.mark.network
def test_encode_decode_shapes():
    codec = MimiCodec(device="cpu")
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    waveform = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    codes = codec.encode(waveform, SAMPLE_RATE)
    assert codes.shape[0] == 1
    assert codes.shape[1] == NUM_CODEBOOKS
    assert codes.min() >= 0
    assert codes.max() < CODEBOOK_SIZE

    decoded = codec.decode(codes)
    assert decoded.ndim == 1
    assert abs(len(decoded) / SAMPLE_RATE - duration_s) < 0.2
