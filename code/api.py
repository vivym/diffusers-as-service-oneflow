import json
from uuid import uuid4
from pathlib import Path
from typing import Optional

import aiofiles
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles

from worker1 import text_to_image_1

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


app.mount("/static", StaticFiles(directory="/static"), name="static")


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)

    result = task_result.result
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            ...

    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": result,
    }


@app.get("/diffusers/text2img")
async def diffusers_text2img(
    prompt: str,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    model_name: str = "stable-diffusion",
):
    if model_name == "stable-diffusion":
        text_to_image = text_to_image_1
    elif model_name == "openjourney":
        text_to_image = None
    elif model_name == "aniplus-v1":
        text_to_image = None
    elif model_name == "anythingv3":
        text_to_image = None
    else:
        return {
            "error": f"invalid model_name: {model_name}"
        }

    task = text_to_image.delay(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        num_outputs=num_outputs,
    )

    return {
        "task_id": task.id,
        "prompt": prompt,
    }


async def save_upload_file(upload_file: UploadFile) -> str:
    file_name = f"{uuid4()}{Path(upload_file.filename).suffix}"
    async with aiofiles.open(f"/uploaded_images/{file_name}", "wb") as f:
        while content := await upload_file.read(4 * 1024):
            await f.write(content)

    return file_name


# @app.post("/diffusers/img2img")
# async def diffusers_img2img(
#     init_image: UploadFile,
#     prompt: str,
#     strength: float = 0.8,
#     guidance_scale: float = 7.5,
#     num_inference_steps: int = 50,
#     width: int = 512,
#     height: int = 512,
#     num_outputs: int = 1,
# ):
#     init_image_path = await save_upload_file(init_image)

#     task = image_to_image.delay(
#         init_image=init_image_path,
#         prompt=prompt,
#         strength=strength,
#         guidance_scale=guidance_scale,
#         num_inference_steps=num_inference_steps,
#         width=width,
#         height=height,
#         num_outputs=num_outputs,
#     )

#     return {
#         "task_id": task.id,
#         "prompt": prompt,
#         "init_image": init_image_path,
#     }
