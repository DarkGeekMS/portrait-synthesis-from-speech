FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /portrait-synthesis-from-speech

ADD . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
