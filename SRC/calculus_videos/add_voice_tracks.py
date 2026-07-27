"""Add the generated female narration WAV tracks to the lesson MP4s."""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "mp4"


def main() -> None:
    for video in sorted(ROOT.glob("*.mp4")):
        audio = video.with_suffix(".wav")
        if not audio.exists():
            continue
        temp = video.with_suffix(".voiced.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
            "-filter_complex", "[1:a]apad[a]", "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", "48", "-movflags", "+faststart", str(temp),
        ]
        subprocess.run(cmd, check=True)
        temp.replace(video)
        print(f"added voice to {video.name}")


if __name__ == "__main__":
    main()
