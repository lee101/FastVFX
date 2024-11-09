# Video AV1 Processing Service

FastAPI + Modal.com service that processes videos using NVIDIA's AV1 encoder.

## Setup

1. Install dependencies:

```bash
pip install uv
python -m uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -r requirements.txt
```

2. Deploy to Modal:
```bash
modal deploy main.py
```

## Usage

The service exposes a POST endpoint at `/process-video` that:
- Accepts MP4 video files
- Applies thresholding and alpha border effects
- Encodes to AV1 using NVIDIA GPU acceleration
- Returns a WebM file

### Parameters
- `video`: MP4 file upload (required)
- `threshold`: Float between 0-1 for video thresholding (default: 0.5)

### Example
```bash
curl -X POST "https://YOUR_MODAL_ENDPOINT/process-video" \
  -F "video=@input.mp4" \
  -F "threshold=0.6" \
  --output processed.webm
```

## Requirements
- NVIDIA GPU with AV1 encoding support (L4 GPU on Modal)
- FFmpeg with NVIDIA acceleration
