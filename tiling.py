import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from pgd import pgd_attack, save_image
import numpy as np
from PIL import Image
import cv2
import dask
import dask.array as da
import os

from transformers import (
    AutoTokenizer,
    CLIPTextModel,
)

global_progress_callback = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_ID = "stabilityai/stable-diffusion-2-1"

# Initialize global model variables
vae = None
tokenizer = None
text_encoder = None
unet = None
noise_scheduler = None
model_loaded = False


# Try to use local model first
def get_model_path():
    """Get the path to the locally downloaded model files"""
    return os.path.join(
        os.path.expanduser("~"), ".ains", "models", "stable-diffusion-2-1"
    )


local_model_path = get_model_path()
if os.path.exists(local_model_path) and os.listdir(local_model_path):
    MODEL_ID = local_model_path
    print(f"Using local model at: {MODEL_ID}")
else:
    # Fallback to online model (will be downloaded at first use)
    MODEL_ID = "stabilityai/stable-diffusion-2-1"


# Initialize models directly
def load_models(progress_callback=None):
    """Load all models during splash screen"""
    global vae, tokenizer, text_encoder, unet, noise_scheduler, global_progress_callback
    global_progress_callback = progress_callback

    if progress_callback:
        progress_callback("Loading model components...")

    # Load all components to device immediately
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float16
    ).to(device)

    if progress_callback:
        progress_callback("VAE model loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    if progress_callback:
        progress_callback("Tokenizer loaded")

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    if progress_callback:
        progress_callback("Text encoder loaded")

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float16
    ).to(device)

    if progress_callback:
        progress_callback("UNet model loaded")

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    if progress_callback:
        progress_callback("All models loaded successfully!")

    # Set models to eval mode (important for inference)
    vae.eval()
    text_encoder.eval()
    unet.eval()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return True


def log_message(message):
    """Global logging function that works outside of classes"""
    print(message)
    if global_progress_callback:
        global_progress_callback(message)


def check_models_exist():
    """Check if models exist on disk without loading them"""
    model_path = os.path.join(
        os.path.expanduser("~"), ".ains", "models", "stable-diffusion-2-1"
    )
    if not os.path.exists(model_path):
        return False

    # Check for essential files
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "unet/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.bin",
        "text_encoder/pytorch_model.bin",
    ]

    for file_path in required_files:
        if not os.path.exists(os.path.join(model_path, file_path)):
            return False

    return True


def process_with_limited_memory(input_image_processor):
    # Move only needed model to GPU as they're needed
    log_message("Moving VAE to GPU for encoding...")
    vae.to(device)

    # Encode image
    encoded_images = input_image_processor._encode_images()

    # Free VAE GPU memory
    vae.to("cpu")
    torch.cuda.empty_cache()

    log_message("Moving text encoder to GPU...")
    text_encoder.to(device)

    # Encode text
    text_embeddings = input_image_processor._encode_prompt()

    # Free text encoder GPU memory
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    log_message("Moving UNet to GPU for perturbation...")
    unet.to(device)

    # Run perturbation
    perturbed_images = input_image_processor._perturb_images(
        encoded_images, text_embeddings
    )

    # Free UNet GPU memory
    unet.to("cpu")
    torch.cuda.empty_cache()

    # Final steps
    log_message("Moving VAE to GPU for decoding...")
    vae.to(device)

    # Decode images
    output_images = input_image_processor._decode_images(perturbed_images)

    # Free all GPU memory
    vae.to("cpu")
    torch.cuda.empty_cache()

    return output_images


def prepare_for_processing():
    """Load models only when needed for processing"""
    global vae, tokenizer, text_encoder, unet, noise_scheduler, model_loaded

    try:
        log_message("Loading AI models for the first time...")

        # Use fp16 precision and memory optimization
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.float16
        )
        log_message("VAE model loaded")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        log_message("Tokenizer loaded")

        try:
            # First try to load to device
            device_to_use = device
            text_encoder = CLIPTextModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
            ).to(device_to_use)
            log_message("Text encoder loaded to GPU")
        except Exception as e:
            # Fall back to CPU if GPU fails
            log_message(
                f"GPU loading failed: {str(e)}. Falling back to CPU for text encoder."
            )
            device_to_use = "cpu"
            text_encoder = CLIPTextModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
            ).to(device_to_use)
            log_message("Text encoder loaded to CPU")

        try:
            # Try loading UNet to GPU
            unet = UNet2DConditionModel.from_pretrained(
                MODEL_ID, subfolder="unet", torch_dtype=torch.float16
            ).to(device_to_use)
            log_message("UNet model loaded to GPU")
        except Exception as e:
            # Fall back to CPU if GPU loading fails
            log_message(f"GPU loading failed for UNet: {str(e)}. Using CPU.")
            unet = UNet2DConditionModel.from_pretrained(
                MODEL_ID, subfolder="unet", torch_dtype=torch.float16
            ).to("cpu")
            log_message("UNet model loaded to CPU")

        # Load scheduler (small model)
        noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        log_message("All models loaded successfully!")

        # Set models to eval mode
        vae.eval()
        text_encoder.eval()
        unet.eval()

        model_loaded = True
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            log_message(
                "GPU OUT OF MEMORY: Try closing other applications or use a smaller batch size"
            )
        log_message(f"Error loading models: {e}")
        return False
    except Exception as e:
        log_message(f"Error loading models: {e}")
        return False


def release_memory():
    """Move models back to CPU after processing to free GPU memory"""
    global vae, text_encoder, unet

    try:
        log_message("Moving AI models back to CPU...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vae.to("cpu")
        text_encoder.to("cpu")
        unet.to("cpu")

        # Force garbage collection
        import gc

        gc.collect()

        log_message("Memory released successfully")
        return True
    except Exception as e:
        log_message(f"Error releasing memory: {e}")
        return False


# Modify TileInator class to use the global model instances
class TileInator:
    def __init__(
        self,
        overlap_size,
        image,
        tile_width,
        tile_height,
        num_cols,
        num_rows,
        intensity,
        prompt,
        filename,
        output_path,
        progress_callback=None,
    ):
        global vae, tokenizer, text_encoder, unet, noise_scheduler, global_progress_callback, device

        # Store reference to global callback
        global_progress_callback = progress_callback
        self.progress_callback = progress_callback

        # Store processing parameters
        self.overlap_size = overlap_size
        self.image = image
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.num_cols = num_cols
        self.num_rows = num_rows

        # Use the global model instances
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.device = device

        self.instance_prompt = prompt
        self.intensity = intensity
        self.filename = filename
        self.output_path = output_path

    def log(self, message):
        """Log message to console and progress callback if available"""
        print(message)
        if self.progress_callback:
            self.progress_callback(message)

    def process_image(self):
        tiles = self._tile_inator()
        self.log("Starting perturbation process...")
        perturbed_tiles = self.lazy_tile_inator(tiles)
        self.log("Finished perturbation. Starting stitching...")
        final_image = self.stitch_perturbed_tiles(
            perturbed_tiles, self.num_rows, self.num_cols
        )
        # Use 4 digits for random integer
        # random_4_digit_integer = random.randint(1000, 9999)
        # Include identifier for attack type in filename
        # output_file_name = f"AdvDiffNoise_GPP_{random_4_digit_integer}.png"
        self.log(f"Saving final stitched image to {self.filename}...")
        base_filename = os.path.basename(self.filename)
        full_output_path = os.path.join(self.output_path, f"protected_{base_filename}")
        final_image.save(full_output_path)
        self.log("Processing complete.")

        return full_output_path

    # _tile_inator remains the same as your latest version (with RGB conversion)
    def _tile_inator(self):
        # --- Ensure input image is RGB ---
        if self.image.mode != "RGB":
            self.log(f"Converting input image from {self.image.mode} to RGB.")
            self.image = self.image.convert("RGB")

        image_np = np.array(self.image)
        h, w, c = image_np.shape

        if c != 3:
            raise ValueError(f"Image conversion failed. Expected 3 channels, got {c}")

        base_height, rem_height = divmod(h, self.num_rows)
        chunks_height = [base_height + 1] * rem_height + [base_height] * (
            self.num_rows - rem_height
        )
        base_width, rem_width = divmod(w, self.num_cols)
        chunks_width = [base_width + 1] * rem_width + [base_width] * (
            self.num_cols - rem_width
        )

        self.log("Creating Dask array for tiling...")
        da_image = da.from_array(image_np, chunks=(chunks_height, chunks_width, 3))
        self.log("Computing tiles from Dask array...")
        tiles = list(dask.compute(*da_image.to_delayed().flatten()))
        self.log(f"Generated {len(tiles)} tiles.")

        # --- Optional: Save initial tiles ---
        # save_dir = os.path.join("resources", "images", "initial_tiles")
        # os.makedirs(save_dir, exist_ok=True)
        # print(f"Saving initial tiles to {save_dir}...")
        # for i, tile_array in enumerate(tiles):
        #     try:
        #         if tile_array.shape[-1] != 3:
        #             continue
        #         tile_array_uint8 = np.clip(tile_array, 0, 255).astype(np.uint8)
        #         tile_img = Image.fromarray(tile_array_uint8)
        #         tile_filename = os.path.join(save_dir, f"tile_{i:03d}.png")
        #         tile_img.save(tile_filename)
        #     except Exception as e:
        #         print(f"Error saving initial tile {i}: {e}")
        # print("Finished saving initial tiles.")
        # --- End saving initial tiles ---

        return tiles  # List of NumPy arrays (RGB, uint8)

    def lazy_tile_inator(self, tiles):
        perturbed_tiles = []
        num_tiles = len(tiles)

        # Prepare prompt encoding once (assuming same prompt for all tiles)
        text_input = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_input = self.tokenizer(  # For classifier-free guidance if needed, though not used in basic loss
                "",
                padding="max_length",
                max_length=text_input.input_ids.shape[-1],
                return_tensors="pt",
            )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        # For basic MSE loss, only text_embeddings are needed
        # text_embeddings = torch.cat(
        # [uncond_embeddings, text_embeddings]
        # )  # If doing CFG

        perturbation_parameters = []
        if self.intensity == 1:
            perturbation_parameters.append(0.04)
            perturbation_parameters.append(0.004)
            perturbation_parameters.append(30)
        elif self.intensity == 2:
            perturbation_parameters.append(0.06)
            perturbation_parameters.append(0.006)
            perturbation_parameters.append(40)
        else:
            perturbation_parameters.append(0.09)
            perturbation_parameters.append(0.009)
            perturbation_parameters.append(50)

        for i, tile_np_rgb in enumerate(tiles):
            self.log(f"  Perturbing tile {i+1}/{num_tiles}...")
            # tile_np_rgb is already numpy RGB uint8 from _tile_inator

            # Perform the diffusion noise prediction attack
            perturbed_tile_tensor = pgd_attack(
                tile_np_rgb,  # Pass the numpy tile (RGB, uint8)
                self.unet,
                text_embeddings,  # Pass pre-computed embeddings
                self.noise_scheduler,
                self.vae,
                self.device,
                epsilon=perturbation_parameters[
                    0
                ],  # Increased epsilon to match Anti-DreamBooth (5e-2)
                alpha=perturbation_parameters[
                    1
                ],  # Adjusted alpha (proportionally or match Anti-DreamBooth if specified, using 1/10th of epsilon here)
                num_iterations=perturbation_parameters[
                    2
                ],  # Keep iterations or adjust as needed
                progress_callback=self.progress_callback,
            )
            perturbed_tiles.append(
                perturbed_tile_tensor
            )  # Append the tensor in [-1, 1] range

            # Optional: Save individual perturbed tiles (adjust save_image if needed)
            # save_dir_adv = os.path.join("resources", "images", "perturbed_tiles_diff")
            # os.makedirs(save_dir_adv, exist_ok=True)
            # save_image(perturbed_tile_tensor, os.path.join(save_dir_adv, f"perturbed_tile_{i:03d}.png"))

        return perturbed_tiles  # List of tensors in [-1, 1] range

    # stitch_perturbed_tiles needs to handle input tensors in [-1, 1] range
    def stitch_perturbed_tiles(self, perturbed_tiles, num_rows, num_cols):
        if not perturbed_tiles or len(perturbed_tiles) != num_rows * num_cols:
            raise ValueError("Invalid number of tiles provided.")

        processed_tiles_np = []
        for tile_tensor in perturbed_tiles:  # Input is now a tensor in [-1, 1]
            # --- Convert tile tensor [-1, 1] to NumPy array [0, 255] uint8 ---
            if isinstance(tile_tensor, torch.Tensor):
                # De-normalize from [-1, 1] to [0, 1]
                tile_tensor_0_1 = tile_tensor * 0.5 + 0.5
                # Permute, detach, cpu, scale, convert type
                tile_np = tile_tensor_0_1.squeeze(0).permute(1, 2, 0).cpu().numpy()
                tile_np = np.clip(tile_np, 0, 1)  # Clip just in case
                tile_np = (tile_np * 255).astype(np.uint8)
            else:
                # This case should ideally not happen if lazy_tile_inator returns tensors
                raise TypeError(f"Expected a torch.Tensor, got {type(tile_tensor)}")

            # --- Ensure 3 channels (RGB) ---
            # (Should be guaranteed by the process, but keep checks)
            if tile_np.ndim == 2:
                tile_np = cv2.cvtColor(tile_np, cv2.COLOR_GRAY2RGB)
            elif tile_np.shape[2] == 4:
                tile_np = cv2.cvtColor(tile_np, cv2.COLOR_RGBA2RGB)
            elif tile_np.shape[2] != 3:
                raise ValueError(f"Tile has unsupported channels: {tile_np.shape[2]}")
            processed_tiles_np.append(tile_np)
            # --- End Tile Conversion ---

        # --- Stitching using hstack and vstack (robust version - remains the same) ---
        rows_of_images = []
        tile_idx = 0
        # ... (rest of the robust stitching logic using processed_tiles_np) ...
        # ... (it should work correctly with the uint8 numpy arrays) ...
        for r in range(num_rows):
            current_row_tiles = []
            max_row_height = 0
            for c in range(num_cols):
                tile_np = processed_tiles_np[tile_idx]
                current_row_tiles.append(tile_np)
                max_row_height = max(max_row_height, tile_np.shape[0])
                tile_idx += 1

            padded_row_tiles = []
            for tile in current_row_tiles:
                h, w, _ = tile.shape
                if h != max_row_height:
                    resized_tile = cv2.resize(
                        tile, (w, max_row_height), interpolation=cv2.INTER_LANCZOS4
                    )
                    padded_row_tiles.append(resized_tile)
                else:
                    padded_row_tiles.append(tile)

            try:
                row_image = np.hstack(padded_row_tiles)
                rows_of_images.append(row_image)
            except ValueError as e:
                self.log(f"Error stacking tiles horizontally in row {r}: {e}")
                for i, t in enumerate(padded_row_tiles):
                    self.log(f"  Tile {i} shape: {t.shape}")
                raise e

        final_rows = []
        if not rows_of_images:  # Handle case with no rows
            return Image.new("RGB", (1, 1))  # Return a dummy 1x1 black image

        max_overall_width = max(row.shape[1] for row in rows_of_images)

        for i, row_img in enumerate(rows_of_images):
            h, w, _ = row_img.shape
            if w != max_overall_width:
                self.log(
                    f"Warning: Resizing row {i} horizontally. Expected {max_overall_width}, got {w}."
                )
                resized_row = cv2.resize(
                    row_img, (max_overall_width, h), interpolation=cv2.INTER_LANCZOS4
                )
                final_rows.append(resized_row)
            else:
                final_rows.append(row_img)

        try:
            stitched_image_np = np.vstack(final_rows)
        except ValueError as e:
            self.log(f"Error stacking rows vertically: {e}")
            for i, row_img in enumerate(final_rows):
                self.log(f"  Row {i} shape: {row_img.shape}")
            raise e

        stitched_image_pil = Image.fromarray(stitched_image_np)
        return stitched_image_pil

    def _group_by_wavefront_inator(self, tiles):
        path = os.path.join("resources", "images", "group")
        os.makedirs(path, exist_ok=True)
        grouped_tiles = []
        for i in range(self.num_cols + self.num_rows - 1):
            inner_path = os.path.join("resources", "images", "group", f"wave_{i}")
            os.makedirs(inner_path, exist_ok=True)
            wave = []
            for row in range(max(0, i - self.num_cols + 1), min(i + 1, self.num_rows)):
                col = i - row
                index = row * self.num_cols + col
                tile_array = tiles[index]
                wave.append(tiles[index])
                tile_img = Image.fromarray(tile_array.astype(np.uint8))
                tile_img.save(os.path.join(inner_path, f"tile_{index}.png"))
            grouped_tiles.append(wave)

        return grouped_tiles

    # to do: implement perturbation and attaching of context edges
    def _perturb_tiles_inator(self, grouped_tiles):
        edges = []

        edges_dir = os.path.join("resources", "images", "edgesAndTile")
        os.makedirs(edges_dir, exist_ok=True)

        # Process each wave
        for wave_idx, wave in enumerate(grouped_tiles):
            wave_edges = []

            # Base case: first wave with single tile
            if wave_idx == 0 and len(wave) == 1:
                self.log("Processing first wave with single tile")
                # Convert tile to format expected by pgd_attack
                first_tile = np.array(wave[0])
                # Convert RGB to BGR for PGD attack (since it uses OpenCV)
                first_tile = cv2.cvtColor(first_tile, cv2.COLOR_RGB2BGR)
                # Apply PGD attack
                perturbed_first_tile = pgd_attack(first_tile, vae, device)
                # Process first tile specially if needed
                wave_edges.append(
                    self.extract_tile_edges(perturbed_first_tile, self.overlap_size)
                )
            else:
                # Normal processing for other waves
                edges_to_attach = edges[wave_idx - 1]
                for tile_idx, tile in enumerate(wave):
                    wave_edges.append(self.extract_tile_edges(tile, self.overlap_size))

                    wave_dir = os.path.join(edges_dir, f"wave_{wave_idx}")
                    os.makedirs(wave_dir, exist_ok=True)
                    # Save the current tile
                    tile_img = None
                    if isinstance(tile, np.ndarray):
                        tile_img = Image.fromarray(tile.astype(np.uint8))
                    elif isinstance(tile, torch.Tensor):
                        tile_array = tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        tile_img = Image.fromarray((tile_array * 255).astype(np.uint8))
                    else:
                        tile_img = Image.fromarray(np.array(tile).astype(np.uint8))

                    tile_img.save(os.path.join(wave_dir, f"tile_{tile_idx}.png"))

                    # Save the edges to attach
                    if tile_idx < len(edges_to_attach):
                        edge_dir = os.path.join(wave_dir, f"edges_for_tile_{tile_idx}")
                        os.makedirs(edge_dir, exist_ok=True)

                        for i, edge in enumerate(edges_to_attach[tile_idx]):
                            edge_type = "right" if i == 0 else "bottom"
                            edge_img = Image.fromarray(edge.astype(np.uint8))
                            edge_img.save(
                                os.path.join(edge_dir, f"{edge_type}_edge.png")
                            )

                    # get adjacent edges
                    # curr_edges = edges_to_attach[i]
                    # stitch edges to current tile in wave
                    # self.stitch_edges_to_tile(curr_edges, wave[i])

                # for t in range(len(wave)):
                #     wave_edges.append(
                #         self.extract_tile_edges(wave[t], self.overlap_size)
                #     )

            edges.append(wave_edges)

        # edges_dir = os.path.join("resources", "images", "edges")
        # os.makedirs(edges_dir, exist_ok=True)

        # wave_count = 0
        # for wave in edges:
        #     wave_dir = os.path.join(edges_dir, f"wave_{wave_count}")
        #     os.makedirs(wave_dir, exist_ok=True)

        #     tile_count = 0
        #     pair_counter = 0
        #     for edg_pair in wave:
        #         edg_pair_dir = os.path.join(wave_dir, f"edge_pair_{pair_counter}")
        #         os.makedirs(edg_pair_dir, exist_ok=True)
        #         for edg in edg_pair:
        #             tile_edge = Image.fromarray(edg.astype(np.uint8))
        #             tile_edge.save(os.path.join(edg_pair_dir, f"tile_{tile_count}.png"))
        #             tile_count += 1
        #         pair_counter += 1
        #     wave_count += 1

    def extract_tile_edges(self, tile, overlap_size):
        overlap_size = int(overlap_size)
        # Handle PyTorch tensor from pgd_attack
        if isinstance(tile, torch.Tensor):
            # Convert tensor to numpy array (same conversion as in save_image)
            tile = tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
            tile = (tile * 255).astype(np.uint8)

        h, w = tile.shape[:2]

        # Right edge
        if w > overlap_size:
            right_edge = tile[:, w - overlap_size :].copy()
        else:
            right_edge = tile.copy()

        # Bottom edge
        if h > overlap_size:
            bottom_edge = tile[h - overlap_size :, :].copy()
        else:
            bottom_edge = tile.copy()

        return [right_edge, bottom_edge]

    def stitch_edges_to_tile(self, edges, tile):
        """
        Attach edges to a tile with smooth blending.

        Parameters:
        - edges: Array containing [right_edge, bottom_edge] or just one edge
        - tile: The target tile to attach edges to

        Returns:
        - Modified tile with edges stitched to it
        """
        # Convert tile to numpy if it's a tensor
        if isinstance(tile, torch.Tensor):
            tile = tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
            tile = (tile * 255).astype(np.uint8)

        # Make a copy of the tile to avoid modifying the original
        modified_tile = np.array(tile).copy()

        # Get tile dimensions
        h, w = modified_tile.shape[:2]
        overlap_size = int(self.overlap_size)

        # Case 1: Two edges - attach both right edge to left side and bottom edge to top side
        if len(edges) == 2:
            right_edge, bottom_edge = edges

            # Attach right edge to left side with alpha blending
            if right_edge.shape[0] == h:  # Make sure heights match
                # Create alpha values for gradual blending (0 at edge, 1 toward center)
                alpha = np.linspace(0, 1, overlap_size)[np.newaxis, :, np.newaxis]
                # Blend the overlap region
                modified_tile[:, :overlap_size] = (
                    1 - alpha
                ) * right_edge + alpha * modified_tile[:, :overlap_size]

            # Attach bottom edge to top side with alpha blending
            if bottom_edge.shape[1] == w:  # Make sure widths match
                # Create alpha values for gradual blending (0 at edge, 1 toward center)
                alpha = np.linspace(0, 1, overlap_size)[:, np.newaxis, np.newaxis]
                # Blend the overlap region
                modified_tile[:overlap_size, :] = (
                    1 - alpha
                ) * bottom_edge + alpha * modified_tile[:overlap_size, :]

        # Case 2: Only right edge
        elif len(edges) == 1 and edges[0].shape[0] == h:  # Right edge (height matches)
            right_edge = edges[0]
            # Create alpha values for gradual blending
            alpha = np.linspace(0, 1, overlap_size)[np.newaxis, :, np.newaxis]
            # Blend the overlap region
            modified_tile[:, :overlap_size] = (
                1 - alpha
            ) * right_edge + alpha * modified_tile[:, :overlap_size]

        # Case 3: Only bottom edge
        elif len(edges) == 1 and edges[0].shape[1] == w:  # Bottom edge (width matches)
            bottom_edge = edges[0]
            # Create alpha values for gradual blending
            alpha = np.linspace(0, 1, overlap_size)[:, np.newaxis, np.newaxis]
            # Blend the overlap region
            modified_tile[:overlap_size, :] = (
                1 - alpha
            ) * bottom_edge + alpha * modified_tile[:overlap_size, :]

        return modified_tile


# class WavefrontTiling:
#     def __init__(self, overlap_size=10):
#         self.overlap_size = overlap_size
#         self.processed_edges = {}  # Store edge information

#     def process_image(self, image, rows, cols):
#         # Main function to process image using wavefront tiling
#         height, width = image.height, image.width
#         tile_height = height // rows
#         tile_width = width // cols

#         # Initialize result image
#         result = np.zeros((height, width, 3), dtype=np.uint8)

#         # Process tiles in wavefront pattern
#         for r in range(rows):
#             for c in range(cols):
#                 print(f"Processing tile at position ({r}, {c})")

#                 # Calculate tile boundaries without overlap first
#                 x1_base = c * tile_width
#                 y1_base = r * tile_height
#                 x2_base = min((c + 1) * tile_width, width)
#                 y2_base = min((r + 1) * tile_height, height)

#                 # Calculate extended boundaries with overlap
#                 x1 = max(0, x1_base - (self.overlap_size if c > 0 else 0))
#                 y1 = max(0, y1_base - (self.overlap_size if r > 0 else 0))
#                 x2 = x2_base
#                 y2 = y2_base

#                 # Extract tile with overlap
#                 tile = image.crop((x1, y1, x2, y2))

#                 # Apply edge context if available
#                 tile = self._apply_edge_context(tile, r, c)

#                 # Process tile using PGD attack
#                 processed_tile = self._process_tile(tile)

#                 # Store edge information for next wave
#                 self._store_edge_info(processed_tile, r, c)

#                 # Remove overlap regions for final placement
#                 processed_tile = self._remove_overlap(processed_tile, r, c)

#                 # Calculate final placement dimensions
#                 place_height = y2_base - y1_base
#                 place_width = x2_base - x1_base

#                 # Ensure processed tile matches placement dimensions
#                 if processed_tile.shape[:2] != (place_height, place_width):
#                     processed_tile = processed_tile[:place_height, :place_width]

#                 # Place processed tile in result
#                 result[y1_base:y2_base, x1_base:x2_base] = processed_tile
#         return Image.fromarray(result)

#     def _apply_edge_context(self, tile, row, col):
#         """Apply edge context from previous tiles"""
#         # Convert PIL Image to numpy array for processing
#         tile_np = np.array(tile)

#         # Apply top edge context if available
#         key = f"{row-1}_{col}"  # Top neighbor
#         if key in self.processed_edges:
#             print("Top edge context applicable. Applying...")
#             top_edge = self.processed_edges[key]["bottom_edge"]
#             # Create alpha values for smooth blending
#             alpha = np.linspace(0, 1, self.overlap_size)[:, np.newaxis, np.newaxis]
#             # Blend the overlapping region
#             overlap_region = (1 - alpha) * top_edge + alpha * tile_np[
#                 : self.overlap_size
#             ]
#             tile_np[: self.overlap_size] = overlap_region

#         # Apply left edge context if available
#         key = f"{row}_{col-1}"  # Left neighbor
#         if key in self.processed_edges:
#             print("Left edge context applicable. Applying...")
#             left_edge = self.processed_edges[key]["right_edge"]
#             # Create alpha values for smooth blending
#             alpha = np.linspace(0, 1, self.overlap_size)[np.newaxis, :, np.newaxis]
#             # Blend the overlapping region
#             overlap_region = (1 - alpha) * left_edge + alpha * tile_np[
#                 :, : self.overlap_size
#             ]
#             tile_np[:, : self.overlap_size] = overlap_region

#         return Image.fromarray(tile_np.astype(np.uint8))

#     def _process_tile(self, tile):
#         """Process single tile using PGD attack"""
#         # Convert tile to format expected by pgd_attack
#         tile_np = np.array(tile)

#         # Convert RGB to BGR for PGD attack (since it uses OpenCV)
#         tile_np = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)

#         # Apply PGD attack
#         processed = pgd_attack(tile_np, vae, device)

#         # Convert back to numpy array
#         processed_np = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         processed_np = (processed_np * 255).astype(np.uint8)

#         return processed_np

#     def _store_edge_info(self, processed_tile, row, col):
#         """Store edge information for next wave"""
#         key = f"{row}_{col}"
#         self.processed_edges[key] = {
#             "right_edge": processed_tile[:, -self.overlap_size :],
#             "bottom_edge": processed_tile[-self.overlap_size :, :],
#         }

#     def _remove_overlap(self, tile, row, col):
#         """Remove overlap regions from processed tile"""
#         h, w = tile.shape[:2]
#         start_y = self.overlap_size if row > 0 else 0
#         start_x = self.overlap_size if col > 0 else 0
#         return tile[start_y:, start_x:]
