import io
import hashlib
import json
from typing import Optional

from celery import Celery
from pymongo import MongoClient

client = MongoClient('mongodb://root:example@mongo:27017/')
db = client.stable_diffusion

app = Celery(
    __name__,
    broker="redis://redis4",
    backend="redis://redis4",
)

pipe = None

def get_pipe():
    global pipe

    if pipe is not None:
        return pipe

    import oneflow as torch
    from diffusers import (
        OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
        OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
    )

    model_id = "Linaqruf/anything-v3.0"

    dpm_solver = DPMSolverMultistepScheduler.from_config(model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=True,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=dpm_solver,
    )

    pipe = pipe.to("cuda")

    return pipe


@app.task
def text_to_image_4(
    prompt: str,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
) -> str:
    pipe = get_pipe()

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_outputs,
    ).images

    paths = []
    for image in images:
        try:
            m = hashlib.sha1()
            with io.BytesIO() as memf:
                image.save(memf, "JPEG")
                data = memf.getvalue()
                m.update(data)

            file_name = f"{m.hexdigest()}.jpg"
            with open(f"/generated_images/{file_name}", "wb") as f:
                f.write(data)
            paths.append(file_name)
        except Exception as e:
            print(f"Postprocess error: {e}")
            paths.append(None)

    results = json.dumps(paths)

    db.text_to_image.insert_one({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "width": width,
        "height": height,
        "num_outputs": num_outputs,
        "results": results,
        "model_name": "Linaqruf/anything-v3.0",
    })

    return paths
