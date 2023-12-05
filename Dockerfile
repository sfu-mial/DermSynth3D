FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel


RUN echo $CUDA_HOME
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH
# ENV CUDA_HOME /usr/local/cuda
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/:$LD_LIBRARY_PATH
# ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#
ENV DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000
ARG USER=developer
ARG GROUP=$USER

ENV FORCE_CUDA=1
RUN echo $(nvcc --version)



# Install necessary packages
RUN --mount=type=cache,target=/var/cache/apt apt update && apt install -y --no-install-recommends \
    sudo \
    git \
    wget \
    bzip2 \
    ca-certificates \
    libx11-6 \
    python3-opencv \
    vim \
    && rm -rf /var/lib/apt/lists/*

## Create a non-root user and group
RUN addgroup --gid $GID $GROUP
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USER && \
    adduser $USER sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# RUN useradd -D -mU ${USER} --uid=${UID}
# Run as this user from now on
USER $USER:$GID

# Install Miniconda
WORKDIR /home/$USER
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/$USER/miniconda/bin:$PATH

# Set up conda environment
COPY dermsynth3d.yml .
RUN conda env create -f dermsynth3d.yml && conda clean -afy
ENV CONDA_DEFAULT_ENV=dermsynth3d
ENV CONDA_PREFIX=/home/$USER/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH


RUN echo "source activate $(head -1 dermsynth3d.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /home/$USER/miniconda/envs/$(head -1 dermsynth3d.yml | cut -d' ' -f2)/bin:$PATH

# Copy code
COPY data /demo_data
# COPY . /home/$USER/DermSynth3D


# Test imports
# RUN git clone --recurse-submodules https://github.com/sfu-mial/DermSynth3D.git
#, "python", "scripts/gen_data.py"]
WORKDIR /home/$USER/DermSynth3D
