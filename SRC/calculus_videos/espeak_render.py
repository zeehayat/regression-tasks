"""Offline female narration renderer using the installed espeak-ng library."""
from __future__ import annotations

import ctypes
import re
import wave
from pathlib import Path


LIB = ctypes.CDLL("libespeak-ng.so.1")
AUDIO_OUTPUT_RETRIEVAL = 1
espeak_Initialize = LIB.espeak_Initialize
espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
espeak_Initialize.restype = ctypes.c_int
LIB.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
LIB.espeak_SetVoiceByName.restype = ctypes.c_int
LIB.espeak_SetSynthCallback.argtypes = [ctypes.c_void_p]
LIB.espeak_SetSynthCallback.restype = ctypes.c_int
LIB.espeak_Synth.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
LIB.espeak_Synth.restype = ctypes.c_int
LIB.espeak_Synchronize.argtypes = []
LIB.espeak_Synchronize.restype = ctypes.c_int
LIB.espeak_Terminate.argtypes = []


def clean_script(text: str) -> str:
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)
    text = text.replace("\n", " ")
    text = text.replace("Δ", "delta ").replace("∇", "gradient ").replace("≈", "approximately")
    text = re.sub(r"[*_`#]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize(text: str, out: Path) -> None:
    samples: list[bytes] = []
    CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

    @CALLBACK
    def callback(wav, numsamples, events):
        if numsamples > 0 and wav:
            samples.append(ctypes.string_at(wav, numsamples * 2))
        return 0

    sample_rate = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 500, None, 0)
    if sample_rate <= 0:
        raise RuntimeError("espeak-ng could not initialize")
    try:
        # en+f3 is a clear female English voice in the installed espeak voice set.
        if LIB.espeak_SetVoiceByName(b"en+f3") != 0:
            LIB.espeak_SetVoiceByName(b"en-us+f3")
        LIB.espeak_SetSynthCallback(callback)
        data = text.encode("utf-8")
        if LIB.espeak_Synth(data, len(data) + 1, 0, 0, 0, 0, None, None) != 0:
            raise RuntimeError("espeak-ng synthesis failed")
        LIB.espeak_Synchronize()
    finally:
        LIB.espeak_Terminate()
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(samples))


def main() -> None:
    root = Path(__file__).resolve().parent
    mp4 = root / "mp4"
    for script in sorted(mp4.glob("*.md")):
        if script.name == "README.md":
            continue
        out = mp4 / script.with_suffix(".wav").name
        synthesize(clean_script(script.read_text(encoding="utf-8")), out)
        print(f"rendered {out.name}")


if __name__ == "__main__":
    main()
