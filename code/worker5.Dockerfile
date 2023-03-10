FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /code

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

CMD ["celery", "-A", "worker5", "worker", "-P", "solo", "--loglevel", "INFO"]
