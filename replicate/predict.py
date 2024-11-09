from cog import BasePredictor, Input, Path
import tempfile
import torch
from torchaudio.io import StreamReader
import numpy as np
import subprocess as sp


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory"""
        pass

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        threshold: float = Input(description="Threshold value", default=0.5)
    ) -> Path:
        # Use /dev/shm for RAM disk storage
        with tempfile.NamedTemporaryFile(suffix='.mp4',
                                         delete=False,
                                         dir="/dev/shm") as output_file:
            output_path = output_file.name

        # Setup video reader with CUDA acceleration
        video_stream = StreamReader(str(video))

        video_ext = str(video).lower()
        if video_ext.endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
            decoder = "h264_cuvid"
        elif video_ext.endswith(".gif"):
            decoder = "gif"
        elif video_ext.endswith((".flv", ".f4v")):
            decoder = "flv"
        elif video_ext.endswith(".wmv"):
            decoder = "wmv3"
        elif video_ext.endswith((".m4v", ".3gp")):
            decoder = "mpeg4"
        elif video_ext.endswith((".ts", ".mts")):
            decoder = "h264"
        else:
            decoder = "h264"  # Fallback decoder
        video_stream.add_video_stream(1, decoder=decoder, hw_accel="cuda:0")

        # Get dimensions from first chunk
        for i, (chunk, ) in enumerate(video_stream.stream()):
            if i == 0:
                frame_height, frame_width = chunk.shape[-2:]
                break

        # Setup FFMPEG pipe with AV1 NVENC
        command = [
            'ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-vcodec',
            'rawvideo', '-pix_fmt', 'rgb24', '-s',
            f'{frame_width}x{frame_height}', '-r', '24', '-i', '-', '-an',
            '-c:v', 'av1_nvenc', '-preset', 'p7', '-rc', 'vbr', '-cq', '19',
            '-pix_fmt', 'yuv420p', '-gpu', '0', '-b:v', '0', '-movflags',
            '+faststart', output_path
        ]

        pipe = sp.Popen(command, stdin=sp.PIPE)
        device = torch.device('cuda')

        # Process frames
        for chunk, in video_stream.stream():
            chunk = chunk.to(device)
            thresholded = (chunk > (threshold * 255)).float() * 255

            frame_rgb = thresholded.cpu().numpy().astype(np.uint8)
            pipe.stdin.write(frame_rgb.tobytes())

        pipe.stdin.close()
        pipe.wait()

        return Path(output_path)
