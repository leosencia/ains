from PIL import Image


def determine_tiling_dimensions(image):
    """
    Prompt the user for tiling intensity (1-3) and calculate the number of rows and columns
    based on the input image size. Intensity 1 results in fewer tiles, while intensity 3
    results in more tiles.
    """
    # while True:
    #     try:
    #         intensity = int(input("Enter tiling intensity (1-3): "))
    #         if intensity in [1, 2, 3]:
    #             break
    #         else:
    #             print("Please enter a valid choice: 1, 2, or 3.")
    #     except ValueError:
    #         print("Please enter an integer.")
    intensity = 2

    # Map intensity to tiling dimensions: using powers of 2 for demonstration
    rows = max(1, int((image.height * intensity) / 500))
    cols = max(1, int((image.width * intensity) / 500))
    print(f"Tiling will be done using {rows} rows and {cols} columns.")
    return rows, cols


def compute_tile_size(image, rows, cols):
    """
    Compute individual tile dimensions given the image and the desired rows and columns.
    """
    tile_width = image.width // cols
    tile_height = image.height // rows
    print(f"Each tile size: {tile_width}x{tile_height} pixels.")
    return tile_width, tile_height
