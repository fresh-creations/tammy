FROM python:3.8

WORKDIR /work_dir

COPY . .

RUN apt-get -y update
RUN apt-get install -y ffmpeg libsndfile1
RUN pip install .
