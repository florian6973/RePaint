from diffusers import StableDiffusionPipeline
import torch
import PIL
import numpy as np
import torchvision.io as io
import cv2
import numpy as np

from tqdm.auto import tqdm

from repaint_scheduler import get_schedule

orig_img = "sources/sd_gt.png"
mask_img = "sources/sd_mask.png"

resized_mask = cv2.imread(mask_img)
resized_mask = cv2.resize(resized_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2RGBA)
cv2.imwrite("outputs/sd_mask_resized.png", resized_mask)

mask_zero = resized_mask[:, :, 0] == 0
resized_mask[mask_zero] = 0

th_resized_mask = torch.from_numpy(resized_mask).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0

orig_img = io.read_image(orig_img).unsqueeze(0).to("cuda")
mask_img_full = io.read_image(mask_img).unsqueeze(0).to("cuda")
mask_img = torch.ones_like(mask_img_full)
mask_img[mask_img_full == 0] = 0

masked_img = orig_img * mask_img

io.write_png(masked_img[0].cpu(), "outputs/sd_original.png")
io.write_png(mask_img[0].cpu() * 255, "outputs/sd_mask.png")

masked_img = (masked_img.float() / 255 * 2) - 1
masked_img = masked_img.to("cuda")

mask_img_full = (mask_img_full.float() / 255 * 2) - 1
mask_img_full = mask_img_full.to("cuda")

# https://huggingface.co/blog/stable_diffusion
# https://wandb.ai/capecape/ddpm_clouds/reports/Using-Stable-Diffusion-VAE-to-encode-satellite-images--VmlldzozNDA2OTgx

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space. 

model = "runwayml/stable-diffusion-inpainting"
model = "runwayml/stable-diffusion-v1-5"
# model = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained(model, subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")

# from diffusers import LMSDiscreteScheduler
# scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler = PNDMScheduler(num_train_timesteps=1000, beta_start = 0.00085, beta_end = 0.012, beta_schedule="scaled_linear")

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 

with torch.no_grad():
    masked_img_encoded = vae.encode(masked_img).latent_dist.sample() * 0.18215
    mask_img_full_encoded = vae.encode(mask_img_full).latent_dist.sample() * 0.18215

mask_img_encoded = th_resized_mask

# cv2.imwrite("outputs/sd_masked_img_encoded-full.png", masked_img_encoded[0].cpu().permute(1, 2, 0).numpy() * 255)
masked_img_encoded = masked_img_encoded * mask_img_encoded

# cv2.imwrite("outputs/sd_masked_img_encoded.png", mask_img_full_encoded[0].cpu().permute(1, 2, 0).numpy() * 255)
# cv2.imwrite("outputs/sd_masked_img_encoded-1.png", mask_img_encoded[0].cpu().permute(1, 2, 0).numpy() * 255)
cv2.imwrite("outputs/sd_encoded.png", masked_img_encoded[0].cpu().permute(1, 2, 0).numpy() * 255)

prompt = ['burger']

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 25 #100           # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(10)    # Seed generator to create the inital latent noise

batch_size = len(prompt)

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)
scheduler.set_timesteps(num_inference_steps)
# latents = latents * scheduler.init_noise_sigma


timesteps = get_schedule(num_inference_steps, scheduler)
last_timestep = torch.tensor(999).to(torch_device)
for t in tqdm(timesteps):
    noise = torch.randn_like(masked_img_encoded, device=torch_device)
    if t < last_timestep:
        noisy_target = scheduler.add_noise(masked_img_encoded, noise, t)

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = noisy_target * mask_img_encoded + latents * (1 - mask_img_encoded)

    else:
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[last_timestep] if last_timestep >= 0 else scheduler.one
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta = 1 - current_alpha_t

        latents = (current_alpha_t ** 0.5) * latents + (1 - current_alpha_t) ** 0.5 * noise

    last_timestep = t


latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample
print(image.shape)

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [PIL.Image.fromarray(image) for image in images]
pil_images[0].save("outputs/sd_output.png")