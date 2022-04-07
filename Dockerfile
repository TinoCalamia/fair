#FROM python:3.9.10-slim
FROM tensorflow/tensorflow:latest-gpu-jupyter
LABEL maintainer="calamia.tino@gmail.com" service=fair

COPY requirements.txt ./

# Install python packages
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache-dir --upgrade --upgrade-strategy=eager -r requirements.txt
RUN pip install jupyterlab
