FROM oneflowinc/oneflow-sd:cu117

WORKDIR /code

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

CMD ["celery", "-A", "worker4", "worker", "--loglevel", "INFO"]
