import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

orig_img = "sources/sd_gt.png"
mask_img = "sources/sd_mask.png"

init_image = PIL.Image.open(orig_img).convert("RGB").resize((512, 512))
mask_image = PIL.Image.open(mask_img).convert("RGB").resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "juicy burger"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.1, classifier_guidance=7.5).images[0]
image.save("outputs/sd_official_output.png")