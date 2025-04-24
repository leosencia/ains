import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


tile_num = 0


# New pgd_attack targeting diffusion noise prediction
def pgd_attack(
    image_np_rgb,  # Expect NumPy array (H, W, 3) RGB uint8
    unet,
    text_embeddings,  # Pass pre-computed text embeddings
    noise_scheduler,
    vae,
    device,
    epsilon=0.03,  # Epsilon in [-1, 1] range (e.g., 0.05 approx 12/255)
    alpha=0.01,  # Alpha in [-1, 1] range (e.g., 0.01 approx 2.5/255)
    num_iterations=15,
):
    global tile_num
    print(f"  Starting Diffusion PGD attack #{tile_num}...")
    tile_num += 1

    # --- Preprocessing ---
    original_h, original_w = image_np_rgb.shape[:2]

    # Optional: Resize to be multiple of 8 for VAE if needed
    # target_h = ((original_h + 7) // 8) * 8
    # target_w = ((original_w + 7) // 8) * 8
    # if (original_h, original_w) != (target_h, target_w):
    #     image_resized = cv2.resize(image_np_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    # else:
    #     image_resized = image_np_rgb
    # Using original size directly might also work if VAE handles it
    image_resized = image_np_rgb

    # Convert to PIL for ToTensor
    pil_image = Image.fromarray(image_resized)

    # Transform to tensor and normalize to [-1, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ]
    )
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    # Convert input tensor to half precision AFTER moving to device
    img_tensor = img_tensor.half()
    # --- End Preprocessing ---

    # Store original tensor for clamping perturbation
    original_img_tensor = img_tensor.clone().detach()

    # Initialize perturbed image tensor
    perturbed_image = img_tensor.clone().detach().requires_grad_(True)

    # PGD Iterations
    for i in range(num_iterations):
        # --- Diffusion Loss Calculation ---
        # 1. Encode image to latent space
        # Use model's scaling factor
        latents = (
            vae.encode(perturbed_image).latent_dist.sample() * vae.config.scaling_factor
        )

        # 2. Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # 3. Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # 4. Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 5. Predict the noise residual (using pre-computed text embeddings)
        noise_pred = unet(
            noisy_latents.half(), timesteps, text_embeddings.half()
        ).sample

        # 6. Calculate the loss (MSE between predicted noise and actual noise)
        # We want to MAXIMIZE this loss for the attack
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        # --- End Diffusion Loss Calculation ---

        # --- Backpropagate (Maximize Loss) ---
        # No need to zero grad models as they are in eval mode and grads disabled
        # Maximize loss by backpropagating it directly (gradient ascent)
        loss.backward()
        # --- End Backpropagate ---

        # --- Update Perturbed Image ---
        with torch.no_grad():
            # Get gradient w.r.t. the perturbed image
            grad = perturbed_image.grad
            # Ensure gradient is valid before using sign
            if grad is None:
                print(f"Warning: Gradient is None at iteration {i}. Skipping update.")
                continue  # Skip update if grad is None

            # Update image using gradient sign (gradient ascent)
            adv_image_step = perturbed_image.float() + alpha * grad.sign().float()

            # Project perturbation (eta) onto L-infinity ball around original image
            eta = torch.clamp(
                adv_image_step - original_img_tensor.float(), min=-epsilon, max=+epsilon
            )

            # Apply clamped perturbation and clamp result to valid range [-1, 1]
            perturbed_image_float = torch.clamp(
                original_img_tensor.float() + eta, min=-1, max=+1
            )

            # Cast back to half precision for the next iteration
            perturbed_image.data = perturbed_image_float.half()

            # Zero the gradient for the next iteration
            perturbed_image.grad.zero_()
        # --- End Update Perturbed Image ---

        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"    PGD Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")

    # Detach final result
    final_adv_image = perturbed_image.detach()

    # Optional: Resize back if initial resizing was done
    # h_final, w_final = final_adv_image.shape[-2:]
    # if (h_final, w_final) != (original_h, original_w):
    #      final_adv_image = F.interpolate(
    #          final_adv_image,
    #          size=(original_h, original_w),
    #          mode="bilinear", # Use bilinear or bicubic
    #          align_corners=False,
    #      )

    # save perturbed tile
    save_image(final_adv_image, f"adversarial_vae_{tile_num}.png")

    torch.cuda.empty_cache()
    # Return tensor in [-1, 1] range
    return final_adv_image


# --- End new pgd_attack ---


# --- save_image function (adjust for [-1, 1] input range) ---
def save_image(tensor, filename):
    """
    Convert tensor image (range [-1, 1]) to PIL and save.
    """
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)  # Remove batch dim

    # De-normalize from [-1, 1] to [0, 1]
    tensor_0_1 = tensor * 0.5 + 0.5

    # Permute, detach, cpu, clip, scale, convert type
    img = tensor_0_1.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    # Save using PIL
    Image.fromarray(img).save(filename)
    # print(f"Saved image to {filename}") # Keep if desired


# --- End save_image function ---
