from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
import torchvision.io as io
import tqdm

from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np

from repaint_scheduler import get_schedule

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
num_steps = 250
scheduler.set_timesteps(num_steps)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

orig_img = r"sources/ddpm_gt.png"
mask_img = r"sources/ddpm_mask.png"

orig_img = io.read_image(orig_img).unsqueeze(0).to("cuda")
mask_img_full = io.read_image(mask_img).unsqueeze(0).to("cuda")

mask_img = torch.ones_like(mask_img_full)
mask_img[mask_img_full == 0] = 0

masked_img = orig_img * mask_img

io.write_png(masked_img[0].cpu(), "outputs/ddpm_original.png")
io.write_png(mask_img[0].cpu() * 255, "outputs/ddpm_mask.png")

masked_img = (masked_img.float() / 255 - 0.5) * 2

timesteps_full = get_schedule(num_steps, scheduler, resampling=5)

last_timestep = 999
for t in tqdm.tqdm(timesteps_full):
    with torch.no_grad():
        noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
        if t < last_timestep:
            noisy_target = scheduler.add_noise(masked_img, noise, t)

            noisy_residual = model(input, t).sample
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample

            input = noisy_target * mask_img + prev_noisy_sample * (1 - mask_img)     
        else:
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[last_timestep] if last_timestep >= 0 else scheduler.one
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta = 1 - current_alpha_t

            input = (current_alpha_t ** 0.5) * input + (1 - current_alpha_t) ** 0.5 * noise

        last_timestep = t 


image = ((input / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
io.write_png(image[0].cpu(), "outputs/ddpm_output.png")