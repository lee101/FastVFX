from cog import BasePredictor, Input, Path
import tempfile
import torch
import numpy as np
import subprocess as sp
from typing import Optional, Tuple
import subprocess
from banding import quantize_colors

try:
    from torchaudio.io import StreamReader
    HAVE_TORCHAUDIO = True
except ImportError:
    HAVE_TORCHAUDIO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        if not HAVE_TORCHAUDIO:
            raise ImportError("torchaudio is required but not installed")

    def _get_decoder_for_extension(self, video_path: str) -> str:
        """Determine the appropriate decoder based on video extension."""
        video_ext = str(video_path).lower()
        
        decoders = {
            ('.mp4', '.mkv', '.avi', '.mov', '.webm'): 'h264_cuvid',
            ('.gif',): 'gif',
            ('.flv', '.f4v'): 'flv',
            ('.wmv',): 'wmv3',
            ('.m4v', '.3gp'): 'mpeg4',
            ('.ts', '.mts'): 'h264'
        }
        
        for extensions, decoder in decoders.items():
            if video_ext.endswith(extensions):
                return decoder
        
        return 'h264'  # Fallback decoder

    def _setup_ffmpeg_pipe(self, frame_width: int, frame_height: int, output_codec: str) -> subprocess.Popen:
        """Setup FFMPEG pipe with AV1 NVENC or H264 NVENC."""
        base_command = [
            'ffmpeg', '-loglevel', 'error', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{frame_width}x{frame_height}',
            '-r', '24',
            '-i', '-',
            '-an'
        ]

        if output_codec == 'h264_nvenc':
            # H264 specific settings
            encode_settings = [
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-tune', 'hq',
                '-rc:v', 'vbr',
                '-cq:v', '19',
                '-b:v', '0',
                '-maxrate:v', '130M',
                '-profile:v', 'high'
            ]
        else:
            # AV1 settings
            encode_settings = [
                '-c:v', 'av1_nvenc',
                '-preset', 'p7',
                '-rc', 'vbr',
                '-cq', '19',
                '-b:v', '0'
            ]

        command = base_command + encode_settings + [
            '-pix_fmt', 'yuv420p',
            '-gpu', '0',
            '-movflags', '+faststart',
            self.output_path
        ]
        
        return sp.Popen(command, stdin=sp.PIPE)

    def _threshold_frame(self, frame: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply color quantization to a frame."""
        return quantize_colors(frame, num_levels=25)

    def _get_frame_dimensions(self, video_stream) -> Tuple[int, int]:
        """Get frame dimensions from the first video chunk."""
        for chunk, in video_stream.stream():
            return chunk.shape[-2:]
        raise ValueError("Could not read video dimensions")

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        num_levels: int = Input(description="Number of color levels", default=25),
        output_codec: str = Input(description="Output codec", default="h264_nvenc")
    ) -> Path:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.mp4',
                                       delete=False,
                                       dir="/dev/shm") as output_file:
            self.output_path = output_file.name

        # Setup video reader
        video_stream = StreamReader(str(video))
        decoder = self._get_decoder_for_extension(video)
        video_stream.add_video_stream(1, decoder=decoder, hw_accel="cuda:0")

        # Get frame dimensions
        frame_height, frame_width = self._get_frame_dimensions(video_stream)

        # Setup FFMPEG pipe
        pipe = self._setup_ffmpeg_pipe(frame_width, frame_height, output_codec)

        try:
            # Process frames
            for chunk, in video_stream.stream():
                chunk = chunk.to(device)
                banded = self._threshold_frame(chunk, num_levels)
                
                frame_rgb = banded.cpu().numpy().astype(np.uint8)
                pipe.stdin.write(frame_rgb.tobytes())
        finally:
            if pipe.stdin:
                pipe.stdin.close()
            pipe.wait()

        return Path(self.output_path)
