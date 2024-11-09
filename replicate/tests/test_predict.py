import torch
import numpy as np
from pathlib import Path
from predict import Predictor

def test_get_decoder_for_extension():
    predictor = Predictor()
    test_cases = [
        ('video.mp4', 'h264_cuvid'),
        ('video.gif', 'gif'),
        ('video.flv', 'flv'),
        ('video.wmv', 'wmv3'),
        ('video.m4v', 'mpeg4'),
        ('video.ts', 'h264'),
        ('video.unknown', 'h264'),
    ]
    
    for filename, expected_decoder in test_cases:
        assert predictor._get_decoder_for_extension(filename) == expected_decoder

def test_threshold_frame():
    predictor = Predictor()
    # Create a sample frame tensor
    frame = torch.tensor([[[0, 128, 255]]], dtype=torch.float32)
    threshold = 0.5  # 127.5 in 0-255 range
    
    expected = torch.tensor([[[0.0, 128.0, 255.0]]], dtype=torch.float32)
    result = predictor._threshold_frame(frame, threshold)
    
    assert torch.allclose(result, expected)

def test_predict():
    predictor = Predictor()
    video_path = Path('input.mp4')
    threshold = 0.5
    
    # Skip test if input file doesn't exist
    if not video_path.exists():
        print("Skipping test_predict: input.mp4 not found")
        return
        
    output_path = predictor.predict(video_path, threshold)
    
    assert output_path.exists()
    assert output_path.suffix == '.mp4' 