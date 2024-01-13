import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

orig_img = "hamburger2.png"
mask_img = "mask.png"

init_image = PIL.Image.open(orig_img).convert("RGB").resize((512, 512))
mask_image = PIL.Image.open(mask_img).convert("RGB").resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "juicy burger"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.1, classifier_guidance=7.5).images[0]
image.save("output.png")