FROM python:3.8-buster

RUN mkdir -p /app
WORKDIR /app

RUN echo server will be running on # Your Port
COPY ./requirements.txt /app/
RUN /usr/local/bin/python -m pip install --upgrade pip & pip install --no-cache-dir --upgrade -r /app/requirements.txt

ENTRYPOINT ["python"]
CMD ["main.py"]