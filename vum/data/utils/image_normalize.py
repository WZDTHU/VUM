"""
Image normalization presets for MAE vs diffusion.
Control via normalize_type: 'mae' | 'diffusion'.
"""
from torchvision import transforms

# MAE / ImageNet: standard ImageNet normalization
NORMALIZE_MAE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_MAE_STD = (0.229, 0.224, 0.225)

# Diffusion: [-1, 1] range
NORMALIZE_DIFFUSION_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_DIFFUSION_STD = (0.5, 0.5, 0.5)


def get_normalize_transform(normalize_type: str, inplace: bool = False):
    """
    Return a transforms.Normalize for the given type.

    Args:
        normalize_type: 'mae' -> ImageNet norm (0.485/0.229); 'diffusion' -> (0.5, 0.5)
        inplace: whether to normalize in-place (if supported)

    Returns:
        transforms.Normalize
    """
    normalize_type = normalize_type.lower().strip()
    if normalize_type == "mae":
        return transforms.Normalize(
            mean=NORMALIZE_MAE_MEAN,
            std=NORMALIZE_MAE_STD,
            inplace=inplace,
        )
    if normalize_type == "diffusion":
        return transforms.Normalize(
            mean=NORMALIZE_DIFFUSION_MEAN,
            std=NORMALIZE_DIFFUSION_STD,
            inplace=inplace,
        )
    raise ValueError(
        f"normalize_type must be 'mae' or 'diffusion', got: {normalize_type!r}"
    )
