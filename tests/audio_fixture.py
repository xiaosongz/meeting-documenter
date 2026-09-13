"""Generate synthetic audio; tests never require a meeting recording."""

import math
import struct
import wave


def write_tone(path, seconds=3):
    sample_rate = 16000
    samples = (
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
        for i in range(sample_rate * seconds)
    )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"".join(samples))
    return path
