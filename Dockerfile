FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

WORKDIR /opt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y curl git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/comfyanonymous/ComfyUI && \
    pip install --no-cache-dir -r ComfyUI/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir opencv-python

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY i2i_canny_api.json ./
COPY videoproc.py ./
RUN pip install --no-cache-dir -r requirements.txt

VOLUME /input
VOLUME /output

COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
