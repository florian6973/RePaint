from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
import tqdm


# pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipeline.to("cuda")
# plt.imshow(pipeline("An image of a squirrel in Picasso style").images[0])
# plt.savefig("test.png")

from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(250)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

orig_img = r"experiments/image_size_inet/outputs/inet256_thick_middle/its_25_jl_1_js_1/gt/000000.png"
mask_img = r"experiments/image_size_inet/outputs/inet256_thick_middle/its_25_jl_1_js_1/gt_keep_mask/000000.png"

orig_img = np.array(Image.open(orig_img))
mask_img = np.array(Image.open(mask_img))

print(orig_img.shape)
print(orig_img)
print(mask_img.shape)
masked_img = orig_img.copy()
masked_img[mask_img == 0] = 0
plt.imshow(masked_img)
plt.savefig("test.png")


masked_img = torch.from_numpy(masked_img).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0

input = scheduler.add_noise(masked_img, noise, scheduler.timesteps[-1])

input = np.array(input.cpu().permute(0, 2, 3, 1).numpy()[0])
plt.imshow(input)
plt.savefig("test2.png")

mask_th = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).to("cuda").float() / 255.0
print(mask_th.shape)
# exit()
input = noise

# exit()

raw_img = masked_img #torch.from_numpy(masked_img).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0
mask_th = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0

print(raw_img.shape)
print(mask_th.shape)
print(scheduler.timesteps)

# concatenate timesteps and reversed timestep and timestep again
# scheduler.timesteps = scheduler.timesteps + scheduler.timesteps[::-1] + scheduler.timesteps
for t in tqdm.tqdm(scheduler.timesteps):
    with torch.no_grad():
        noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
        noisy_target = scheduler.add_noise(raw_img, noise, t)

        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample

        input = noisy_target * mask_th + prev_noisy_sample * (1 - mask_th)      

        # input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
plt.imshow(image)
plt.savefig("testz.png")