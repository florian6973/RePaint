from io import BytesIO
import torch
import PIL
import requests
from diffusers import RePaintPipeline, RePaintScheduler


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


orig_img = r"experiments/image_size_inet/outputs/inet256_thick_middle/its_25_jl_1_js_1/gt/000000.png"
mask_img = r"experiments/image_size_inet/outputs/inet256_thick_middle/its_25_jl_1_js_1/gt_keep_mask/000000.png"

# img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
# mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

# Load the original image and the mask as PIL images
# original_image = download_image(img_url).resize((256, 256))
# mask_image = download_image(mask_url).resize((256, 256))
original_image = PIL.Image.open(orig_img).resize((256, 256))
mask_image = PIL.Image.open(mask_img).resize((256, 256))

# Load the RePaint scheduler and pipeline based on a pretrained DDPM model
# scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
# pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)

scheduler = RePaintScheduler.from_pretrained("google/ddpm-cat-256")
pipe = RePaintPipeline.from_pretrained("google/ddpm-cat-256", scheduler=scheduler)
pipe = pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(0)
output = pipe(
    image=original_image,
    mask_image=mask_image,
    num_inference_steps=150,
    eta=0.0,
    jump_length=10,
    jump_n_sample=10,
    generator=generator,
)
inpainted_image = output.images[0]
inpainted_image.save("output-r.png")