# Video Thresholding with CUDA Acceleration

This repository contains a Cog-compatible implementation of video thresholding with CUDA acceleration. The model processes video files by applying a threshold filter and outputs the result in AV1 format using NVIDIA's hardware encoder.

## How to use

fork the project and deploy your own version.

create a new model in replicate.com, and in cog.yaml, change the image to your username/repo.

```bash
image: "r8.im/lee101/fast-vfx"
```
Change how you process the video in [predict.py](replicate/predict.py).



then push your changes to replicate.

```bash
cog push
```

then you can use your model with the following command.

```bash
sudo cog predict -i video="https://netwrckstatic.netwrck.com/input2.mp4" -i threshold=0.5
```

## Development

```bash
cd replicate
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```



## deploy

```bash
cog push
```

## Implementation Details

The predictor uses:
- TorchAudio's StreamReader for efficient video decoding
- CUDA-accelerated video processing
- NVIDIA's AV1 encoder (av1_nvenc) for high-quality compression
- RAM disk storage for faster I/O operations

## Key Features

- Hardware-accelerated video decoding with multiple codec support
- CUDA-based thresholding operation
- Efficient AV1 encoding with NVIDIA GPU
- Temporary file handling using RAM disk
- Support for various input formats:
  - MP4, MKV, AVI, MOV, WEBM
  - GIF
  - FLV, F4V
  - WMV
  - M4V, 3GP
  - TS, MTS

## How to deploy

```bash
cog push
```

## Future Improvements

- [ ] Add support for batch processing
- [ ] Implement more video filters
- [ ] Support CPU fallback when GPU is unavailable
- [ ] Add video preprocessing options

## Performance Considerations

The predictor is optimized for GPU execution with:
- CUDA-accelerated video decoding
- In-memory frame processing
- Hardware-accelerated encoding
- RAM disk usage for temporary files

I need to test newer codecs like AV1 etc.

Please help me test and improve this model!

Let me know if you need help with implementation details or have suggestions for improvements!