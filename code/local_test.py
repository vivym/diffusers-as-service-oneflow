import oneflow as torch
from diffusers import (
    OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
)

# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "prompthero/openjourney"

dpm_solver = DPMSolverMultistepScheduler.from_config(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    scheduler=dpm_solver,
)

pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
print("start")
with torch.autocast("cuda"):
    images = pipe(prompt, num_inference_steps=20, num_images_per_prompt=2).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")


prompt = "a photo of an astronaut riding a horse on mars"
print("start")
with torch.autocast("cuda"):
    images = pipe(prompt, num_inference_steps=20, num_images_per_prompt=2).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
