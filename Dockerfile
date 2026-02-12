# Dockerfile for running ROS2 with NVIDIA GPU support
# using this now cuz i think i'll need to use ros2 soon.
FROM althack/ros2:jazzy-full

RUN sudo apt-get update && sudo apt-get install -y \
    libosmesa6-dev libgl1 libglfw3 patchelf \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# let's use conda so we don't fight the original UMI setup
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/mamba && \
    ln -s /opt/mamba/bin/mamba /usr/bin/mamba && \
    rm Miniforge3-$(uname)-$(uname -m).sh

SHELL ["/bin/bash", "-c"]
COPY . .
RUN mamba env create -f conda_environment.yaml
RUN echo "source /opt/mamba/etc/profile.d/conda.sh && conda activate umi" >> ~/.bashrc 
