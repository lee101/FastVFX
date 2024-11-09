import cv2
import torch
import numpy as np
from pathlib import Path
from banding import quantize_colors

def test_quantize_colors():
    # Load test image
    img_path = Path("tests/image.png")
    
        
    # Read image with OpenCV and convert to RGB
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to torch tensor and normalize to [0,255] range
    img_tensor = torch.from_numpy(img).float()
    
    # Apply color quantization
    num_levels = 8
    quantized = quantize_colors(img_tensor, num_levels)
    
    # Convert back to numpy array for saving
    quantized_img = quantized.numpy().astype(np.uint8)
    quantized_img = cv2.cvtColor(quantized_img, cv2.COLOR_RGB2BGR)
    
    # Save output image
    output_path = "quantized_image.png"
    cv2.imwrite(output_path, quantized_img)
    
    # Verify output was created
    assert Path(output_path).exists()
    
    # Basic validation of quantization
    unique_values = torch.unique(quantized)
    assert len(unique_values) <= num_levels * 3  # 3 channels
    assert torch.all(quantized >= 0)
    assert torch.all(quantized <= 255)
