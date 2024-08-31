
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

MAINTAINER "Sandeep Upadhyay<sandeepupadhyay732@gmail.com>"

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates \
    ffmpeg \
    libsm6 \
    libxext6

RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet

RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils


WORKDIR /object-detection

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install base Python requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

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

# remove the existing python3 binary
RUN rm /usr/bin/python3

# createa symantic link from python3.11 to python3
RUN ln -s /usr/bin/python3.11 /usr/bin/python3

# ENTRYPOINT ["python3"]
