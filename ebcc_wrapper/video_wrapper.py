import numpy as np
import subprocess
import tempfile
import os


class FFmpegVideoArrayCompressor:
    """
    Compress / decompress numpy arrays (n, h, w) in [0,1] float32 using FFmpeg.

    Requirements:
      - ffmpeg and ffprobe installed and on PATH.
    """

    def __init__(
        self,
        codec: str = "libx264",
        fps: int = 25,
        crf: int = 23,
        preset: str = "medium",
        container: str = "mp4",
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ):
        self.codec = codec
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.container = container
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def compress(self, arr: np.ndarray) -> bytes:
        """
        arr: (n, h, w), float32 in [0,1]
        returns: video bitstream (bytes)
        """
        if arr.ndim != 3:
            raise ValueError(f"Expected array shape (n, h, w), got {arr.shape}")

        n, h, w = arr.shape

        # ensure float32 in [0,1]
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)

        # to uint8 grayscale
        frames_u8 = (arr * 255.0 + 0.5).astype(np.uint8)  # (n, h, w)
        raw_bytes = frames_u8.tobytes()

        # temp output video file
        suffix = "." + self.container
        out_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        out_path = out_file.name
        out_file.close()

        # ffmpeg: rawvideo from stdin -> encoded video file
        cmd = [
            self.ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "gray",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-an",
            "-vcodec", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "gray",
            out_path,
        ]

        try:
            subprocess.run(
                cmd,
                input=raw_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            # read encoded file as bitstream
            with open(out_path, "rb") as f:
                bitstream = f.read()
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

        return bitstream

    def decompress(self, bitstream: bytes) -> np.ndarray:
        """
        bitstream: bytes as returned by compress()
        returns: (n, h, w) float32 in [0,1]
        """
        if not isinstance(bitstream, (bytes, bytearray)):
            raise TypeError("bitstream must be bytes or bytearray")

        # write bitstream to temp video file
        suffix = "." + self.container
        in_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        in_path = in_file.name
        try:
            in_file.write(bitstream)
            in_file.flush()
        finally:
            in_file.close()

        try:
            # get width, height via ffprobe (super simple CSV output)
            probe_cmd = [
                self.ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                in_path,
            ]
            proc = subprocess.run(
                probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            wh = proc.stdout.decode("utf-8").strip()
            if "x" not in wh:
                raise RuntimeError(f"Could not parse width/height from ffprobe: {wh}")
            w_str, h_str = wh.split("x")
            w, h = int(w_str), int(h_str)

            # decode to raw grayscale frames to stdout
            decode_cmd = [
                self.ffmpeg,
                "-loglevel", "error",
                "-i", in_path,
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                "pipe:1",
            ]
            proc = subprocess.run(
                decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            raw = proc.stdout
            if not raw:
                raise RuntimeError("ffmpeg decoding produced no data")

            frame_size = w * h
            if len(raw) % frame_size != 0:
                raise RuntimeError(
                    f"Raw size {len(raw)} not divisible by frame size {frame_size}"
                )

            n = len(raw) // frame_size
            arr_u8 = np.frombuffer(raw, dtype=np.uint8).reshape((n, h, w))
            arr_f32 = arr_u8.astype(np.float32) / 255.0
            return arr_f32
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)

if __name__ == "__main__":
    n, h, w = 10, 721, 720
    x = np.random.rand(n, h, w).astype(np.float32)

    compressor = FFmpegVideoArrayCompressor(
        codec="libx265",  # or libx265, libvpx-vp9, etc
        # fps=10,
        # crf=23,
        # preset="medium",
        # container="mp4",
    )

    bitstream = compressor.compress(x)
    print("compressed bytes:", len(bitstream))

    x_hat = compressor.decompress(bitstream)
    print("decoded shape:", x_hat.shape)
    print("decoded range:", float(x_hat.min()), float(x_hat.max()))
