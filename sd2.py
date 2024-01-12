from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
im = pipeline("An image of a squirrel in Picasso style").images[0]
im.save("squirrel.png")