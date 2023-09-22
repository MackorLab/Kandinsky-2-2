#!/usr/bin/env python

from __future__ import annotations

import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import spaces
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline

DESCRIPTION = "# Kandinsky 2.2"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
    )
    pipe_prior.to(device)
    pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
    pipe.to(device)
    if USE_TORCH_COMPILE:
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
else:
    pipe_prior = None
    pipe = None


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU
def generate(
    prompt: str,
    negative_prompt: str = "low quality, bad quality",
    seed: int = 0,
    width: int = 768,
    height: int = 768,
    guidance_scale_prior: float = 1.0,
    guidance_scale: float = 4.0,
    num_inference_steps_prior: int = 50,
    num_inference_steps: int = 100,
) -> PIL.Image.Image:
    generator = torch.Generator().manual_seed(seed)
    image_embeds, negative_image_embeds = pipe_prior(
        prompt,
        negative_prompt,
        generator=generator,
        guidance_scale=guidance_scale_prior,
        num_inference_steps=num_inference_steps_prior,
    ).to_tuple()
    image = pipe(
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    return image


examples = [
    "An astronaut riding a horse",
    "portrait of a young woman, blue eyes, cinematic",
    "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting",
    "bird eye view shot of a full body woman with cyan light orange magenta makeup, digital art, long braided hair her face separated by makeup in the style of yin Yang surrealism, symmetrical face, real image, contrasting tone, pastel gradient background",
    "A car exploding into colorful dust",
    "editorial photography of an organic, almost liquid smoke style armchair",
    "birds eye view of a quilted paper style alien planet landscape, vibrant colours, Cinematic lighting",
    "Toy smiling cute octopus in a black hat, sticker",
    "Red sport car, sticker",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Box():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
        with gr.Accordion("Advanced options", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt",
                value="low quality, bad quality",
                max_lines=1,
                placeholder="Enter a negative prompt",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=768,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=768,
            )
            guidance_scale_prior = gr.Slider(
                label="Guidance scale for prior",
                minimum=1,
                maximum=20,
                step=0.1,
                value=1.0,
            )
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=1,
                maximum=20,
                step=0.1,
                value=4.0,
            )
            num_inference_steps_prior = gr.Slider(
                label="Number of inference steps for prior",
                minimum=10,
                maximum=100,
                step=1,
                value=50,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=10,
                maximum=150,
                step=1,
                value=100,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    inputs = [
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        guidance_scale_prior,
        guidance_scale,
        num_inference_steps_prior,
        num_inference_steps,
    ]
    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name="run",
    )
    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
