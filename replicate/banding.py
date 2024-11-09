import torch

def quantize_colors(image: torch.Tensor, num_levels: int = 25) -> torch.Tensor:
    """
    Quantize an input image tensor to a specified number of color levels.
    
    Args:
        image: Input tensor of shape (C, H, W) with values in range [0, 255]
        num_levels: Number of distinct color levels to use (default: 25)
    
    Returns:
        Quantized tensor of same shape as input with values in range [0, 255]
    """
    # Calculate step size between levels
    step = 255.0 / (num_levels - 1)
    
    # Round values to nearest level
    quantized = torch.round(image / step) * step
    
    # Ensure output is in valid range
    quantized = torch.clamp(quantized, 0, 255)
    
    return quantized
