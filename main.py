# filepath: c:\Users\Lawrence\OneDrive - University of the Philippines\Documents\CMSC\4th Year\SP\AINS\codes\main.py
from image_preprocessing import determine_tiling_dimensions, compute_tile_size
from PIL import Image
from tiling import TileInator
import os  # Import the os module
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

# from tiling import WavefrontTiling


def main():
    # Define the directory containing the images
    image_dir = "resources/images/tarkovsky/"

    # Check if the directory exists
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found - {image_dir}")
        return

    print(f"Processing images in directory: {image_dir}")

    # Loop through all files in the directory
    for filename in os.listdir(image_dir):
        # Check if the file is a .png or .jpg image (case-insensitive)
        if filename.lower().endswith((".png", ".jpg")):
            image_path = os.path.join(image_dir, filename)
            print(f"\n--- Processing image: {filename} ---")

            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
                continue  # Skip to the next file if opening fails

            # --- Perform the same operations as before ---
            rows, cols = determine_tiling_dimensions(image)
            tile_width, tile_height = compute_tile_size(image, rows, cols)

            # Ensure overlap_size is an integer
            overlap_size = int(tile_height * 0.1)

            print(
                f"  Tiling with {rows} rows, {cols} columns. Overlap: {overlap_size}px"
            )

            input_image_processor = TileInator(
                overlap_size=overlap_size,
                image=image,
                tile_width=tile_width,
                tile_height=tile_height,
                num_cols=cols,
                num_rows=rows,
            )
            try:
                input_image_processor.process_image()
                print(f"  Finished processing {filename}.")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                # Optionally add more detailed error logging here
            # --- End of operations ---
        else:
            # Optional: Print message for non-image files found
            # print(f"Skipping non-image file: {filename}")
            pass

    print("\n--- Finished processing all images in the directory ---")


if __name__ == "__main__":
    main()
