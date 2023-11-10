# FROM cr.msk.sbercloud.ru/aijcontest_official/fbc3_0:0.1 as base
FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252
USER root
WORKDIR /app

# -- Build, tag, push and run image
# sudo docker build --tag supermachina:0.15 .
# sudo docker tag supermachina:0.15 cr.msk.sbercloud.ru/aijcontest/supermachina:0.15
# sudo docker push cr.msk.sbercloud.ru/aijcontest/supermachina:0.15
# sudo docker run --rm -it supermachina:0.15 -- sh

# -- Build for multi platforms
# sudo docker buildx build --platform linux/amd64 -f ./Dockerfile --tag supermachina:0.2 .

# -- Show and prune Docker cache
# sudo docker system df
# sudo docker builder prune

# -- Show and remove unused images
# sudo docker image ls
# sudo docker image rm supermachina:0.1

# -- Show TOP 20 biggest files and folders
# sudo du -ah / | sort -rh | head -n 20

# -- Show which process occupied some local port
# sudo lsof -i:8888 -P -n | grep LISTEN

# -- Reset GPU
# nvidia-smi --gpu-reset

# -- Show and kill processes using GPU
# lsof | grep /dev/nvidia

# COPY model.gguf /app/model.gguf
# COPY projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind
# COPY .checkpoints/imagebind_huge.pth /app/.checkpoints/imagebind_huge.pth
# COPY ./Llama-2-7B-fp16 /app/Llama-2-7B-fp16

COPY ./mistral /app/mistral
COPY ./lora /app/lora
# COPY ./.checkpoints /app/.checkpoints

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends mc nano git htop lsof make build-essential python3-pip ffmpeg

RUN pip install Pillow
RUN pip install evaluate

RUN pip install requests
RUN pip install sentencepiece
RUN pip install transformers

RUN pip install https://github.com/enthought/mayavi/zipball/master
RUN pip install --upgrade git+https://github.com/lizagonch/ImageBind.git aac_datasets torchinfo

RUN git clone https://github.com/sshh12/multi_token && cd multi_token && pip install -e .
# RUN pip install flash-attn --no-build-isolation

# -- See standard Python libs: https://docs.python.org/3/library/index.html
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt 

USER jovyan
WORKDIR /home/jovyan
