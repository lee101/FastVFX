from typing import Optional
from cog import BasePredictor, Input, Path
import tempfile
import torch
import numpy as np
import subprocess as sp
import subprocess
import cv2
from banding import quantize_colors
import json
from pathlib import Path as PathLib

try:
    from torchaudio.io import StreamReader
    HAVE_TORCHAUDIO = True
except ImportError:
    HAVE_TORCHAUDIO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_available_hw(video_path: str) -> Optional[str]:
    """Get the available hardware acceleration for a video file"""
    # Check if file extension supports hardware acceleration
    video_ext = str(video_path).lower()
    
    # Extensions that support CUDA hardware acceleration
    cuda_extensions = {'.mp4', '.hevc', '.webm', '.h265', '.mkv', '.avi', '.mov'}
    
    # Check if extension supports CUDA
    for ext in cuda_extensions:
        if video_ext.endswith(ext):
            return 'cuda:0'
            
    # Return None for formats without CUDA support
    return None
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
            ('.hevc', '.h265'): 'hevc_cuvid',
            ('.av1', '.avif'): 'av1_cuvid',
            ('.vp8', '.webm'): 'vp8_cuvid',
            ('.vp9',): 'vp9_cuvid',
            ('.gif',): 'gif',
            ('.flv', '.f4v'): 'flv1',
            ('.wmv',): 'wmv', 
            ('.m4v', '.3gp'): 'mpeg4',
            ('.ts', '.mts'): 'h264'
        }
        
        for extensions, decoder in decoders.items():
            if video_ext.endswith(extensions):
                return decoder
        
        return 'h264'  # Fallback decoder

    def _probe_audio_details(self, video_path: str) -> dict:
        """Get detailed audio stream information using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # Find the audio stream
        audio_stream = next((stream for stream in data['streams'] 
                            if stream['codec_type'] == 'audio'), None)
        
        if audio_stream:
            return {
                'start_time': float(audio_stream.get('start_time', 0)),
                'duration': float(audio_stream.get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0))
            }
        return None

    def _probe_video_details(self, video_path: str) -> dict:
        """Get detailed video stream information using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and data['streams']:
            stream = data['streams'][0]
            # Parse frame rate fraction (e.g., "30/1" -> 30)
            r_frame_rate = stream.get('r_frame_rate', '24/1')
            num, den = map(int, r_frame_rate.split('/'))
            frame_rate = num / den
            
            return {
                'codec_name': stream.get('codec_name'),
                'width': int(stream.get('width', 0)),
                'height': int(stream.get('height', 0)),
                'frame_rate': frame_rate
            }
        return {
            'codec_name': 'h264',
            'width': 0,
            'height': 0,
            'frame_rate': 24.0  # fallback
        }

    def _setup_ffmpeg_pipe(self, frame_width: int, frame_height: int, output_codec: str, audio_path: Optional[str] = None) -> subprocess.Popen:
        """Setup FFMPEG pipe using image2pipe for PNG input."""
        # Get input video details including frame rate
        video_details = self._probe_video_details(self.input_video_path)
        frame_rate = video_details['frame_rate']
        print(f"Frame rate: {frame_rate}")
        base_command = [
            'ffmpeg',
            '-loglevel', 'error',
            '-y',
            
            # Input format configuration
            '-f', 'image2pipe',
            '-framerate', str(frame_rate),  # Use the input video's frame rate
            '-i', '-',  # Read from stdin
        ]

        # Add audio input with proper timing based on audio analysis
        if audio_path:
            audio_details = self._probe_audio_details(self.input_video_path)
            if audio_details:
                start_time = audio_details['start_time']
                base_command.extend([
                    '-itsoffset', str(start_time),
                    '-i', audio_path,
                ])

        if output_codec == 'h264_nvenc':
            # H264 specific settings
            encode_settings = [
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-rc', 'vbr',
                '-cq', '19',
                '-b:v', '0',
                '-pix_fmt', 'yuv444p',
                '-profile:v', 'high444p',
                '-tune', 'hq'
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

        # Audio settings
        if audio_path:
            encode_settings.extend([
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-async', '1'    # Enable audio-video sync
            ])
        else:
            encode_settings.extend(['-an'])  # No audio

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

    def _threshold_frame(self, frame: torch.Tensor, num_levels: int) -> torch.Tensor:
        """Apply color quantization to a frame."""
        # return frame
        return quantize_colors(frame, num_levels=num_levels)

    def _probe_audio(self, video_path: str) -> bool:
        """Check if video has audio stream using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        return any(stream['codec_type'] == 'audio' for stream in data['streams'])

    def _extract_audio(self, video_path: str, audio_path: str):
        """Extract audio from video file."""
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # Disable video
            '-acodec', 'copy',  # Copy audio codec without re-encoding
            '-y',  # Overwrite output
            audio_path
        ]
        subprocess.run(cmd, check=True)

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        num_levels: int = Input(description="Number of color levels", default=25),
        output_codec: str = Input(description="Output codec", default="h264_nvenc")
    ) -> Path:
        self.input_video_path = str(video)  # Store input path for audio analysis
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.mp4',
                                       delete=False,
                                       dir="/dev/shm") as output_file:
            self.output_path = output_file.name

        # Check for audio and extract if present
        audio_path = None
        if self._probe_audio(str(video)):
            audio_path = str(PathLib(tempfile.gettempdir()) / "temp_audio.aac")
            self._extract_audio(str(video), audio_path)

        # Setup video reader
        video_stream = StreamReader(str(video))
        decoder = self._get_decoder_for_extension(video)
        print(f"Using decoder: {decoder}")
        
        hw_accel = get_available_hw(str(video))
        video_stream.add_video_stream(1, decoder=decoder, hw_accel=hw_accel)

        pipe = None
        try:
            # Process frames
            for chunk, in video_stream.stream():
                chunk = chunk.to(device)
                
                # Setup FFMPEG pipe on first frame using its dimensions
                if pipe is None:
                    frame_height, frame_width = chunk.shape[-2:]
                    pipe = self._setup_ffmpeg_pipe(frame_width, frame_height, output_codec, audio_path)
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
        
        # Cleanup temporary audio file
        if audio_path and PathLib(audio_path).exists():
            PathLib(audio_path).unlink()

        return Path(self.output_path)
