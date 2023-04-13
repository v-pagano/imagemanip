#!/usr/local/bin/python

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse
import random
import torch
from time import time


parser = argparse.ArgumentParser(description='Setup up your text to image run')
parser.add_argument('--prompt', type=str, required=True, help='Prompt for image creation')
parser.add_argument('--nImages', type=int, required=False, help='How many images do you want to create?', default=5)
parser.add_argument('--steps', type=int, required=False, help='How many inference steps', default=50)
parser.add_argument('--guidance', type=float, required=False, help='Inference guidance (1-10)', default=7.5)
parser.add_argument('--seed', type=int, required=False, help='Random seed to use', default=random.randint(1, 65535))
args = parser.parse_args()


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

image = pipe(prompt = args.prompt, guidance_scale = args.guidance).images[0]
    
image.save(f'{time()}.png')