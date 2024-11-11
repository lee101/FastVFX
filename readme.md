# Video FX Server with CUDA Acceleration

A high-performance video processing server that leverages NVIDIA CUDA acceleration for real-time video effects. Perfect for AI model development and video processing pipelines.

## Key Features

- 🚀 GPU-accelerated video processing using CUDA
- 🎥 Efficient streaming pipeline all in the GPU - decode, process, encode
- 💾 Memory-efficient processing for long videos
- 🔌 Easy deployment on replicate.com
- 🛠️ Extensible base for custom video AI models

## Quick Start

1. Fork this repository
2. Update the model name in `replicate/cog.yaml`:
```bash
image: "r8.im/your-username/your-model-name"
```
3. Deploy to replicate:
```bash
cog push
```

Example: https://replicate.com/lee101/fast-vfx

this uses banding as an example but make any special effect you like :)

## Development

```bash
# Setup development environment
cd replicate
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Build and test locally
sudo cog build
docker run -d -p 5000:5000 --gpus all r8.im/your-username/your-model-name
```

Edit the replicate/predict.py file to customize the video processing

### Testing the API

Basic video processing:
```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{"input": {"video": "https://example.com/video.mp4"}}'
```

With custom parameters:
```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{"input": {
    "video": "https://example.com/input.mp4",
    "num_levels": 25,
    "output_codec": "h264_nvenc"
  }}'
```

Using replicate CLI:
```bash
sudo cog predict -i video="https://example.com/input.mp4" -i threshold=0.5
```

Deploy to replicate
```bash
cog push
```

## Technical Details

- Uses TorchAudio's StreamReader for efficient video decoding
- CUDA-accelerated video processing pipeline
- Supports NVIDIA's hardware encoders (h264_nvenc, av1_nvenc)
- RAM disk storage for optimized I/O operations
- Supports multiple input formats (MP4, MKV, AVI, MOV, WEBM, GIF, etc.)

For detailed implementation and API documentation, see [replicate/readme.md](replicate/readme.md)

## Projects and Plugs
Let me know if theres some good projects that that showcase this tech!

Please support my projects:

- 🎨 [Stable Diffusion Server](https://github.com/lee101/stable-diffusion-server)
- 🌐 [Netwrck - AI Video, Art and Chat Platform](https://netwrck.com)
- ✍️ [Text Generator](https://text-generator.io) - [Text Generator Git Repo](https://github.com/TextGeneratorio/text-generator.io)
- 🎨 [AI Art Generator](https://ebank.nz)

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Share your implementations and use cases
- Please fork!!!

## License

[MIT License](LICENSE)