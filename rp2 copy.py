from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
import torchvision.io as io
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
num_steps = 25
scheduler.set_timesteps(num_steps)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

orig_img = r"experiments/image_size_inet/outputs/inet256_thick_middle/its_25_jl_1_js_1/gt/000000.png"
mask_img = r"experiments/image_size_inet/outputs/inet256_thick_middle/its_25_jl_1_js_1/gt_keep_mask/000000.png"

orig_img = io.read_image(orig_img).unsqueeze(0).to("cuda")
mask_img_full = io.read_image(mask_img).unsqueeze(0).to("cuda")

# masked_img = orig_img.clone()
# masked_img[mask_img_full == 0] = 0

mask_img = torch.ones_like(mask_img_full)
mask_img[mask_img_full == 0] = 0

masked_img = orig_img * mask_img


io.write_png(masked_img[0].cpu(), "original.png")
io.write_png(mask_img[0].cpu() * 255, "mask.png")

masked_img = (masked_img.float() / 255 - 0.5) * 2

# exit()

# orig_img = np.array(Image.open(orig_img))
# mask_img = np.array(Image.open(mask_img))

# print(orig_img.shape)
# print(orig_img)
# print(mask_img.shape)
# masked_img = orig_img.copy()
# masked_img[mask_img == 0] = 0
# plt.imshow(masked_img)
# plt.savefig("test.png")


# masked_img = torch.from_numpy(masked_img).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0

# input = scheduler.add_noise(masked_img, noise, scheduler.timesteps[-1])

# input = np.array(input.cpu().permute(0, 2, 3, 1).numpy()[0])
# plt.imshow(input)
# plt.savefig("test2.png")

# mask_th = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).to("cuda").float() / 255.0
# print(mask_th.shape)
# # exit()
# input = noise

# exit()

# raw_img = masked_img #torch.from_numpy(masked_img).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0
# mask_th = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255.0

# print(raw_img.shape)
# print(mask_th.shape)
# print(scheduler.timesteps)

# half_timesteps = scheduler.timesteps[:len(scheduler.timesteps) // 2]
# half_timesteps_reverse = half_timesteps.flip(0)[1:-1]
# last_half_timestep = scheduler.timesteps[len(scheduler.timesteps) // 2:]
# half_timesteps_reverse = last_half_timestep.flip(0)[1:-1]
# concatenate timesteps and reversed timestep and timestep again
# timesteps_full = torch.cat((scheduler.timesteps, scheduler.timesteps.flip(0)[1:-1], scheduler.timesteps))
# timesteps_full = torch.cat((scheduler.timesteps, half_timesteps_reverse, last_half_timestep)) #,half_timesteps_reverse, last_half_timestep, half_timesteps_reverse, last_half_timestep, half_timesteps_reverse, last_half_timestep, half_timesteps_reverse, last_half_timestep))
# timesteps_full = scheduler.timesteps

ts = []
def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    # _check_times(ts, -1, t_T)

    # ts = [5*i-1 for i in range(25, 0, -1)] + [-1]
    # print(ts)

    return ts

ts = get_schedule_jump(num_steps, 1, 10, 10, 1, 1, 1, 1, 100000000)[:-1]
ts = (np.array(ts)*scheduler.config.num_train_timesteps/num_steps).astype(int)
print(ts)
np.savetxt("ts.txt", scheduler.alphas_cumprod, fmt="%f")
# exit()
timesteps_full = torch.from_numpy(ts)#.to("cuda")
print(timesteps_full)
print(scheduler.alphas_cumprod)
# exit()
# exit()
# input = noise
# for t in tqdm.tqdm(scheduler.timesteps):
last_timestep = 1000
for t in tqdm.tqdm(timesteps_full):
    with torch.no_grad():
        noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
        if t < last_timestep:
            noisy_target = scheduler.add_noise(masked_img, noise, t)

            noisy_residual = model(input, t).sample
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample

            input = noisy_target * mask_img + prev_noisy_sample * (1 - mask_img)     
        else:
            # input = scheduler.add_noise(input, noise, t)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[last_timestep] if last_timestep >= 0 else scheduler.one
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta = 1 - current_alpha_t
            print(t//20, current_beta)

            input = (current_alpha_t ** 0.5) * input + (1 - current_alpha_t) ** 0.5 * noise

        last_timestep = t 

        # input = prev_noisy_sample

image = ((input / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
io.write_png(image[0].cpu(), "output.png")

# image = (input / 2 + 0.5).clamp(0, 1)
# image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
# image = Image.fromarray((image * 255).round().astype("uint8"))
# plt.imshow(image)
# plt.savefig("testz.png")