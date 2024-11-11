# Video FX Implementation Details

Technical documentation for the Video FX Server implementation. This guide covers the internal workings, API, and deployment specifics.

## API Reference

### Prediction Endpoint

```bash
POST /predictions
```

Parameters:
```json
{
  "input": {
    "video": "string (URL)",
    "threshold": "float (0-1)",
    "output_codec": "string (h264_nvenc|av1_nvenc)",
    "num_levels": "integer (optional)"
  }
}
```

Example requests:
```bash
# Basic usage
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{"input": {"video": "https://example.com/video.mp4"}}'

# With all parameters
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{"input": {
    "video": "https://netwrckstatic.netwrck.com/input2.mp4",
    "num_levels": 25,
    "threshold": 0.5,
    "output_codec": "h264_nvenc"
  }}'
```

## Implementation Architecture

### Video Processing Pipeline

1. **Decoding**: TorchAudio StreamReader decodes video frames efficiently
2. **Processing**: CUDA-accelerated frame processing
3. **Encoding**: Hardware-accelerated encoding using NVIDIA codecs

### Storage Optimization

- Uses RAM disk for temporary files
- Streaming architecture prevents memory accumulation
- Efficient cleanup of intermediate files

## Development Guide

### Local Development Setup

```bash
cd replicate
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Testing

```bash
# Install test dependencies
pip install pytest

# Run tests
PYTHONPATH=. pytest
```

### Docker Development

```bash
# Build container
sudo cog build

# Run interactive shell
docker run -i -t -u root --gpus all --entrypoint=/bin/bash r8.im/your-username/your-model-name

# Test container
docker run -d -p 5000:5000 --gpus all r8.im/your-username/your-model-name
```

### Replicate Deployment

1. Update your model name in cog.yaml:
```yaml
image: "r8.im/your-username/your-model-name"
```

2. Push to replicate:
```bash
cog push
```

3. Test deployment:
```bash
sudo cog predict -i video="https://netwrckstatic.netwrck.com/input2.mp4" -i threshold=0.5
```

## Supported Formats

### Input Formats
- Video: MP4, MKV, AVI, MOV, WEBM
- Animation: GIF
- Legacy: FLV, F4V, WMV, M4V, 3GP
- Stream: TS, MTS

### Output Codecs
- h264_nvenc (Default)
- av1_nvenc (Better compression)

## Performance Optimization

Current optimizations:
- CUDA-accelerated processing
- Hardware video decoding
- In-memory frame processing
- RAM disk I/O optimization

Planned improvements:
- [ ] Batch processing support
- [ ] Additional video filters
- [ ] CPU fallback mode
- [ ] Video preprocessing options
- [ ] Codec comparison benchmarks

## Troubleshooting

Common issues and solutions:
1. CUDA out of memory: Adjust batch size or processing resolution
2. Codec not found: Ensure NVIDIA drivers are properly installed
3. Performance issues: Check GPU utilization and memory usage

## Contributing

Please help improve this implementation by:
- Testing different codecs and configurations
- Reporting performance metrics
- Suggesting optimization strategies
- Adding new video processing features

For questions or contributions, please open an issue or pull request in the repository.