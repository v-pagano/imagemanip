#!/opt/conda/bin/python

import numpy as np
import torch
from torch import autocast
from slugify import slugify
from time import time

from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from diffusion_utils import stable_diffusion_preprocess_image

import argparse
import random

parser = argparse.ArgumentParser(description='Setup up your image to image run')
parser.add_argument('--prompt', type=str, required=True, help='Prompt for image creation')
parser.add_argument('--image', type=str, required=True, help='Image to use as baseline')
parser.add_argument('--nImages', type=int, required=False, help='How many images do you want to create?', default=5)
parser.add_argument('--steps', type=int, required=False, help='How many inference steps', default=50)
parser.add_argument('--guidance', type=float, required=False, help='Inference guidance (1-20)', default=7.5)
parser.add_argument('--strength', type=float, required=False, help='How much noise to introduce (0-1)', default=0.6)
parser.add_argument('--seed', type=int, required=False, help='Random seed to use', default=random.randint(1, 65535))
args = parser.parse_args()

model_path = 'models/stable-diffusion-v1.4'

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    scheduler=scheduler,
    revision='fp16',
    torch_dtype=torch.float16,
).to('cuda')

start_image_preprocessed = stable_diffusion_preprocess_image(args.image)
start_image_preprocessed

seed_generator = torch.Generator('cuda').manual_seed(args.seed)

prompt      = args.prompt
repetitions = args.nImages

with autocast('cuda'):
    for r in range(args.nImages):
        image = model(
            prompt = args.prompt,
            init_image = start_image_preprocessed,
            num_inference_steps=args.steps,
            strength = args.strength,
            guidance_scale = args.guidance,
            generator = generator,
        )
        image = image['images'][0]
        image.show()
        image.save(f'/scratch/vpagano/imagemanip/{time()}_{slugify(args.prompt[:100])}.png')
