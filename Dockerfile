FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
#
ARG UID
ARG GID
ARG USER=developer
ARG GROUP=developer
#
## Create a non-root user and group
RUN addgroup --gid ${GID} ${GROUP} \
 && adduser --disabled-password --gecos '' --uid ${UID} --gid ${GID} ${USER}

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    wget \
    bzip2 \
    ca-certificates \
    libx11-6 \
    python3-opencv \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
USER ${USER}
WORKDIR /home/${USER}
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/${USER}/miniconda/bin:$PATH

# Set up conda environment
COPY dermsynth3d.yml .
RUN conda env create -f dermsynth3d.yml && conda clean -afy
ENV CONDA_DEFAULT_ENV=dermsynth3d
ENV CONDA_PREFIX=/home/${USER}/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Activate conda environment
RUN echo "conda activate dermsynth3d" >> ~/.bashrc

# Mount data drive
USER root
RUN mkdir /data && chown ${USER}:${GROUP} /data
USER ${USER}
VOLUME /data

# Copy code
COPY . .

RUN conda init bash

RUN conda init
#CMD source ~/.bashrc

#RUN conda activate dermsynth3d

# Test imports
RUN python -c 'import torch; print(torch.__version__)'

