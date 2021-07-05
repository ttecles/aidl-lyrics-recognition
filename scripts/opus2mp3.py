import functools
import multiprocessing as mp
from pathlib import Path
import subprocess


def convert(source: Path, dest_folder: Path):
    dest = dest_folder / source.name
    output = subprocess.run(f"ffmpeg -i {source} -acodec libmp3lame {dest}", shell=True,
                            capture_output=False, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    print("File", source, "converted in ", dest)


if __name__ == '__main__':
    opus = Path("/home/joan/AIDL/aidl-lyrics-recognition/data/opus")
    audio = Path("/home/joan/AIDL/aidl-lyrics-recognition/data/audio")

    opus_files = opus.glob("*.mp3")
    converted_files = [f.name for f in audio.glob("*.mp3")]
    opus_files = [f for f in opus_files if f.name not in converted_files]
    print("Files to be converted (AFTER filter):", len(opus_files))
    # opus_files = [Path("/home/joan/AIDL/aidl-lyrics-recognition/data/opus/fffdba22ae1647a2910541b4d4ec3bed.mp3")]
    # converted_files = [Path("/home/joan/AIDL/aidl-lyrics-recognition/data/audio/0a0c413b5290497c96d5327e2ef2ad8d.mp3")]
    convert_func = functools.partial(convert, dest_folder=audio)
    with mp.Pool(10) as p:
        p.map(convert_func, opus_files)
