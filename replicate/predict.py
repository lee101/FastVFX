from cog import BasePredictor, Input, Path
import tempfile
import torch
import numpy as np
import subprocess as sp
from typing import Optional, Tuple
import subprocess
import cv2
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
        """Setup FFMPEG pipe using image2pipe for PNG input."""
        base_command = [
            'ffmpeg',
            '-loglevel', 'error',
            '-y',
            
            # Input format configuration
            '-f', 'image2pipe',
            '-framerate', '24',
            '-i', '-',  # Read from stdin
            
            # Disable audio
            '-an'
        ]

        if output_codec == 'h264_nvenc':
            # H264 specific settings
            encode_settings = [
                '-c:v', 'h264_nvenc',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv444p'
            ]
        else:
            # AV1 settings
            encode_settings = [
                '-c:v', 'av1_nvenc',
                '-preset', 'p7',
                '-rc', 'vbr',
                '-cq', '19',
                '-b:v', '0',
                '-pix_fmt', 'yuv444p'
            ]

        command = base_command + encode_settings + [
            '-gpu', '0',
            '-movflags', '+faststart',
            self.output_path
        ]
        
        return sp.Popen(command, stdin=sp.PIPE)

    def _frame_to_png_bytes(self, frame: np.ndarray) -> bytes:
        """Convert frame to PNG bytes using OpenCV."""
        success, encoded_image = cv2.imencode('.png', frame)
        if not success:
            raise RuntimeError("Failed to encode frame as PNG")
        return encoded_image.tobytes()

    def _threshold_frame(self, frame: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply color quantization to a frame."""
        # return frame
        return quantize_colors(frame, num_levels=25)

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

        pipe = None
        try:
            # Process frames
            for chunk, in video_stream.stream():
                chunk = chunk.to(device)
                
                # Setup FFMPEG pipe on first frame using its dimensions
                if pipe is None:
                    frame_height, frame_width = chunk.shape[-2:]
                    pipe = self._setup_ffmpeg_pipe(frame_width, frame_height, output_codec)
                print(chunk.shape)
                banded = self._threshold_frame(chunk, num_levels)
                print(banded.shape)
                print(banded.device)
                print(banded.dtype)
                frame_rgb = banded.cpu().numpy().astype(np.uint8)
                # Convert from [1, 3, H, W] to [H, W, 3] format for OpenCV
                frame_rgb = frame_rgb.squeeze(0).transpose(1, 2, 0)
                # Convert BGR to RGB since OpenCV uses BGR by default
                # Convert from RGB to YUV colorspace for proper color handling
                # frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YUV)
                # Convert back to RGB 
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_YUV2RGB)
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                # Convert frame to PNG bytes and write to pipe
                png_bytes = self._frame_to_png_bytes(frame_rgb)
                pipe.stdin.write(png_bytes)
        finally:
            if pipe and pipe.stdin:
                pipe.stdin.close()
            if pipe:
                pipe.wait()

        return Path(self.output_path)
