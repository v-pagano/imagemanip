#!/opt/conda/bin/python

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from time import time
from slugify import slugify
import argparse
import random

parser = argparse.ArgumentParser(description='Setup up your text to image run')
parser.add_argument('--prompt', type=str, required=True, help='Prompt for image creation')
parser.add_argument('--nImages', type=int, required=False, help='How many images do you want to create?', default=5)
parser.add_argument('--steps', type=int, required=False, help='How many inference steps', default=50)
parser.add_argument('--guidance', type=float, required=False, help='Inference guidance (1-10)', default=7.5)
parser.add_argument('--seed', type=int, required=False, help='Random seed to use', default=random.randint(1, 65535))
args = parser.parse_args()

model = StableDiffusionPipeline.from_pretrained(
    '/scratch/vpagano/imagemanip/project/models/stable-diffusion-v1.4',
    revision='fp16',
    torch_dtype=torch.float16
)
model.to('cuda')

seed_generator = torch.Generator('cuda').manual_seed(args.seed)

with autocast('cuda'):
    for r in range(args.nImages):
        output = model(
            prompt=args.prompt,
            generator=seed_generator,
            num_inference_steps=args.steps, # diffusion iterations
            guidance_scale=args.guidance,     # adherence to text, default 7.5
            width=512,
            height=512,
        )

        image = output['images'][0]
        image.save(f'{time()}_{slugify(args.prompt[:100])}.png')
