ARG CUDA_VERSION=12.6
ARG TORCH_BASE=full

FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}

LABEL maintainer="XXXXRT"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS"

ARG CUDA_VERSION=12.6

ENV CUDA_VERSION=${CUDA_VERSION}

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}

RUN bash Docker/miniconda_install.sh

COPY extra-req.txt /workspace/GPT-SoVITS/

COPY requirements.txt /workspace/GPT-SoVITS/

COPY install.sh /workspace/GPT-SoVITS/

# 接受 Conda 服务条款
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV TERM=xterm

RUN bash Docker/install_wrapper.sh

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

RUN conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS
