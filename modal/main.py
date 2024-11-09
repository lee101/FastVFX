import modal

# Create Modal image with required dependencies
image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11").pip_install(
    "fastapi[standard]",
    "python-multipart", 
    "torch",
    "torchaudio",
    "torchvision",
    find_links="https://download.pytorch.org/whl/cu124"
).run_commands(
    # Install build dependencies
    "apt-get update && apt-get install -y build-essential cmake yasm pkg-config git curl",
    "apt-get install -y libx264-dev libx265-dev libnuma-dev libvpx-dev libmp3lame-dev libopus-dev libaom-dev",
    # Build NVIDIA headers
    "git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git",
    "cd nv-codec-headers && make install && cd .. && rm -rf nv-codec-headers",
    # Build FFmpeg 6
    "git clone --branch release/6.1 --depth 1 https://git.ffmpeg.org/ffmpeg.git ffmpeg",
    "cd ffmpeg && ./configure --enable-nonfree --enable-cuda-nvcc--enable-gpl --enable-libx264 --enable-libx265 --enable-nvenc --enable-cuvid --enable-libvpx --enable-libopus --enable-libmp3lame --enable-libaom --enable-libsvtav1 --prefix=/usr/local",
    "cd ffmpeg && make -j$(nproc) && make install", 
    "cd .. && rm -rf ffmpeg",
    # Update library path
    "ldconfig"
)

app = modal.App("video-av1-app", image=image)


with app.image.imports():
    from fastapi import FastAPI, UploadFile, File, Response
    from fastapi.responses import FileResponse
    from torchaudio.io import StreamReader
    import numpy as np
    import torch
    import tempfile
import os
import subprocess as sp


web_app = FastAPI()

@app.function(gpu="L4")
async def process_video(input_path: str, output_path: str, threshold: float = 0.5):
    # Setup video reader with CUDA acceleration
    video_stream = StreamReader(input_path)
    video_stream.add_video_stream(1, decoder="h264_cuvid", hw_accel="cuda:0")

    # Get dimensions from first chunk
    for i, (chunk,) in enumerate(video_stream.stream()):
        if i == 0:
            frame_height, frame_width = chunk.shape[-2:]
            break

    # Setup FFMPEG pipe with AV1 NVENC
    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',
        # Input
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{frame_width}x{frame_height}',
        '-r', '24',
        # Output
        '-i', '-',
        '-an',
        '-c:v', 'av1_nvenc',
        '-preset', 'p7',
        '-rc', 'vbr',
        '-cq', '19',
        '-pix_fmt', 'yuv420p',
        '-gpu', '0',
        '-b:v', '0',
        '-movflags', '+faststart',
        output_path
    ]

    pipe = sp.Popen(command, stdin=sp.PIPE)
    device = torch.device('cuda')

    # Simplified frame processing
    for chunk, in video_stream.stream():
        chunk = chunk.to(device)
        thresholded = (chunk > (threshold * 255)).float() * 255
        
        # Write frame without alpha
        frame_rgb = thresholded.cpu().numpy().astype(np.uint8)
        pipe.stdin.write(frame_rgb.tobytes())

    pipe.stdin.close()
    pipe.wait()

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/process-video")
async def handle_video(
    video: UploadFile = File(...),
    threshold: float = 0.5
):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_file:
        input_path = input_file.name
        content = await video.read()
        input_file.write(content)

    output_path = input_path.replace('.mp4', '_processed.mp4')
    await process_video.remote(input_path, output_path, threshold)

    # Read the processed video into memory
    with open(output_path, 'rb') as f:
        video_data = f.read()

    # Clean up files
    os.unlink(input_path)
    os.unlink(output_path)

    # Return video data directly in response
    return Response(
        content=video_data,
        media_type='video/mp4',
        headers={'Content-Disposition': 'attachment; filename="processed.mp4"'}
    )

if __name__ == "__main__":
    app.deploy("webapp")
