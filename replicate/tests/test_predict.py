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
        ('video.flv', 'flv1'),
        ('video.wmv', 'wmv'),
        ('video.m4v', 'mpeg4'),
        ('video.ts', 'h264'),
        ('video.unknown', 'h264'),
    ]
    
    for filename, expected_decoder in test_cases:
        assert predictor._get_decoder_for_extension(filename) == expected_decoder

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


def test_predict_with_real_video():
    """Test the predict function with a real video file"""
    predictor = Predictor()
    video_path = Path('input.mp4')
    
    output_path = predictor.predict(video_path, 5, "h264_nvenc") 
    print(f"Output path: {output_path}")
    # Save output video to disk
    output_dir = Path('tests')
    output_dir.mkdir(exist_ok=True)
    test_output_path = output_dir / 'test_output.mp4'
    
    # Copy output file to test directory
    import shutil
    shutil.copy(output_path, test_output_path)
    assert output_path.exists()
    assert output_path.suffix == '.mp4'
    
def test_predict_cant_keep_getting_away():
    """Test prediction on cant-keep-getting-away.mp4"""
    predictor = Predictor()
    video_path = Path('cant-keep-getting-away.mp4')
    
    output_path = predictor.predict(video_path, 10, "h264_nvenc")
    print(f"Output path: {output_path}")
    
    # Save output video to disk
    output_dir = Path('tests')
    output_dir.mkdir(exist_ok=True)
    test_output_path = output_dir / 'test_output_cant_keep_getting_away.mp4'
    
    # Copy output file to test directory
    import shutil
    shutil.copy(output_path, test_output_path)
    
    assert output_path.exists()
    assert output_path.suffix == '.mp4'
    assert test_output_path.exists()
    assert test_output_path.stat().st_size > 0, "Output file is empty"

def test_predict_all_test_data():
    """Test prediction on all video files in test_data directory"""
    predictor = Predictor()
    test_data_dir = Path('test_data')
    output_dir = Path('tests/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all video files from test_data directory
    video_extensions = {'.mp4', '.gif', '.flv', '.wmv', '.m4v', '.ts', '.mkv', '.avi', '.mov', '.webm', '.hevc', '.h265', '.av1', '.avif', '.vp8', '.vp9'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(test_data_dir.glob(f'*{ext}'))
    
    assert len(video_files) > 0, "No video files found in test_data directory"
    
    for video_path in video_files:
        print(f"Processing {video_path}")
        output_path = predictor.predict(
            video=video_path,
            num_levels=8,
            output_codec="h264_nvenc"
        )
        
        # Copy to test outputs directory with original filename
        test_output_path = output_dir / f"output_{video_path.name}.mp4"
        import shutil
        shutil.copy(output_path, test_output_path)
        
        # Basic validation
        assert output_path.exists()
        assert output_path.suffix == '.mp4'
        assert test_output_path.exists()
        assert test_output_path.stat().st_size > 0, f"Output file {test_output_path} is empty"