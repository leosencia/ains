import math


def determine_tiling_dimensions(image, segmentation):
    """
    Calculate the number of rows and columns dynamically based on:
    1. Image dimensions
    2. Segmentation level (1-3)
    3. Target tile sizes that work well with the model
    """
    # Get image dimensions
    width, height = image.width, image.height

    # Calculate the total number of pixels in the image
    total_pixels = width * height

    # Target tile sizes (in pixels) for different segmentation levels
    if segmentation == 1:  # Low segmentation - larger tiles
        target_tile_pixels = 128 * 1024  # ~128K pixels per tile
    elif segmentation == 2:  # Medium segmentation
        target_tile_pixels = 65 * 1024  # ~65K pixels per tile (~256x256)
    else:  # High segmentation - smaller tiles
        target_tile_pixels = 32 * 1024  # ~32K pixels per tile

    # Calculate approximate number of tiles needed
    target_tiles = max(1, total_pixels / target_tile_pixels)

    # Find a balanced grid size (trying to keep tiles somewhat square)
    aspect_ratio = width / height

    # Calculate rows and columns to maintain aspect ratio
    cols = int(round(math.sqrt(target_tiles * aspect_ratio)))
    rows = int(round(target_tiles / cols))

    # Ensure at least one row and column
    rows = max(1, rows)
    cols = max(1, cols)

    print(f"Image {width}x{height}, Segmentation level {segmentation}")
    print(f"Tiling using {rows} rows and {cols} columns")

    return rows, cols


def compute_tile_size(image, rows, cols):
    """
    Compute individual tile dimensions given the image and the desired rows and columns.
    """
    tile_width = image.width // cols
    tile_height = image.height // rows
    print(f"Each tile size: {tile_width}x{tile_height} pixels.")
    return tile_width, tile_height
