import torch
import numpy as np
from pathlib import Path
from predict import Predictor
import os
import cv2


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
    
        
    output_path = predictor.predict(video_path, 25, "h264_nvenc")
    # Save output video to disk
    output_dir = Path('tests')
    output_dir.mkdir(exist_ok=True)
    test_output_path = output_dir / 'test_output.mp4'
    
    # Copy output file to test directory
    import shutil
    shutil.copy(output_path, test_output_path)
    assert output_path.exists()
    assert output_path.suffix == '.mp4' 


# def test_predict_with_real_video():
#     """Test the predict function with a real video file"""

    
#     # Create a simple test video
#     output_dir = Path('tests')
#     output_dir.mkdir(exist_ok=True)
    
#     test_video_path = output_dir / 'test_input.mp4'
    
#     # Create a simple video with color gradients
#     width, height = 320, 240
#     fps = 30
#     duration = 2  # seconds
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(str(test_video_path), fourcc, fps, (width, height))
    
#     try:
#         # Generate frames with changing colors
#         for i in range(fps * duration):
#             # Create a gradient frame
#             frame = np.zeros((height, width, 3), dtype=np.uint8)
#             frame[:, :, 0] = np.linspace(0, 255, width)  # Blue gradient
#             frame[:, :, 1] = np.linspace(0, 255, height)[:, None]  # Green gradient
#             frame[:, :, 2] = (i / (fps * duration) * 255)  # Red changing over time
#             out.write(frame)
#     finally:
#         out.release()
    
#     # Now test the predictor
#     predictor = Predictor()
#     num_levels = 8  # Use 8 distinct color levels
    
#     try:
#         # Process the video
#         output_path = predictor.predict(test_video_path, num_levels, "h264_nvenc")
        
#         # Verify the output
#         assert output_path.exists()
#         assert output_path.suffix == '.mp4'
        
#         # Verify the output video can be opened and has frames
#         cap = cv2.VideoCapture(str(output_path))
#         try:
#             ret, frame = cap.read()
#             assert ret, "Could not read frame from output video"
#             assert frame.shape == (height, width, 3), "Output video has incorrect dimensions"
#         finally:
#             cap.release()
            
#     finally:
#         # Cleanup
#         if test_video_path.exists():
#             test_video_path.unlink()
#         if output_path.exists():
#             output_path.unlink()
#         if output_dir.exists():
#             output_dir.rmdir()