FROM pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime

WORKDIR /code

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

CMD ["celery", "-A", "worker5", "worker", "--loglevel", "INFO"]
