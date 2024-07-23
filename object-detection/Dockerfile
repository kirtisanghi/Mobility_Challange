## base image

FROM python:3.11.9

MAINTAINER "Sandeep Upadhyay<sandeepupadhyay732@gmail.com>"
WORKDIR /object-detection
COPY requirements.txt requirements.txt
RUN pip install --user --requirement requirements.txt

RUN git clone https://github.com/ultralytics/ultralytics
WORKDIR /object-detection/ultralytics
RUN git reset --hard 2071776a3672eb835d7c56cfff22114707765ac
RUN wget https://gist.githubusercontent.com/Y-T-G/8f4fc0b78a0a559a06fe84ae4f359e6e/raw/17b1407fefeac86d089c4cf14f174c8bb44948af/add_head.patch
RUN git apply add_head.patch
RUN pip install --editable .

WORKDIR /application
COPY app.py /application/
COPY ml /application/ml/
COPY models /application/models/

ENTRYPOINT ["app.py"]

#FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04 as base
#
#RUN apt-get update &&  apt-get install -y python3-pip
#
#WORKDIR /object-detection
#
## Install base Python requirements
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt
#
#RUN git clone https://github.com/ultralytics/ultralytics
#WORKDIR /object-detection/ultralytics
#RUN git reset --hard 2071776a3672eb835d7c56cfff22114707765ac
#RUN wget https://gist.githubusercontent.com/Y-T-G/8f4fc0b78a0a559a06fe84ae4f359e6e/raw/17b1407fefeac86d089c4cf14f174c8bb44948af/add_head.patch
#RUN git apply add_head.patch
#RUN pip install --editable .
#
#WORKDIR /application
#COPY app.py /application/
#COPY ml /application/ml/
#COPY models /application/models/
