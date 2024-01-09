FROM python:3.11

WORKDIR /opt
RUN git clone https://github.com/comfyanonymous/ComfyUI && \
    pip install --no-cache-dir -r ComfyUI/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY i2i_canny_api.json ./
COPY videoproc.py ./
RUN pip install --no-cache-dir -r requirements.txt

VOLUME /input
VOLUME /output

COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
