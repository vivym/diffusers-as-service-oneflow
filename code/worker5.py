import io
import hashlib
import json
import sys
from typing import Optional

import requests
from celery import Celery
from PIL import Image

try:
    import torch
    from RealESRGAN import RealESRGAN
except:
    ...


app = Celery(
    __name__,
    broker="redis://redis5",
    backend="redis://redis5",
)

model_x2 = None
model_x4 = None


def get_model_x2():
    global model_x2

    if model_x2 is not None:
        return model_x2

    model_x2 = RealESRGAN(torch.device("cuda"), scale=2)
    model_x2.load_weights("/weights/RealESRGAN_x2.pth")

    return model_x2


def get_model_x4():
    global model_x4

    if model_x4 is not None:
        return model_x4

    model_x4 = RealESRGAN(torch.device("cuda"), scale=4)
    model_x4.load_weights("/weights/RealESRGAN_x4.pth")

    return model_x4


@app.task
def super_resolution(
    image_url: str,
    scale: int,
) -> str:
    if scale == 2:
        model = get_model_x2()
    elif scale == 4:
        model = get_model_x4()
    else:
        raise NotImplementedError(scale)

    import numpy as np

    rsp = requests.get(image_url)
    image = Image.open(io.BytesIO(rsp.content)).convert("RGB")
    sr_image = model.predict(np.asarray(image))

    m = hashlib.sha1()
    with io.BytesIO() as memf:
        sr_image.save(memf, "JPEG")
        data = memf.getvalue()
        m.update(data)

    file_name = f"{m.hexdigest()}.jpg"
    with open(f"/generated_images/{file_name}", "wb") as f:
        f.write(data)

    return file_name
