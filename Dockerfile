FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install Miniconda
RUN apt-get update && apt-get install -y curl bzip2 build-essential && \
    curl -o /tmp/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    ln -s /opt/conda/bin/conda /usr/bin/conda && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA false
ENV EGL_PLATFORM=surfaceless

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"


# Create hugging_face environment and install dependencies
RUN bash -c "source /opt/conda/bin/activate && \
    conda create -n hugging_face python=3.10 -y && \
    source /opt/conda/bin/activate hugging_face && \
    pip install gradio pygltflib open3d==0.18.0 laspy laszip"

# Create Open3D ML environment
RUN bash -c "source /opt/conda/bin/activate && \
    conda create -n o3d_ml python=3.8 -y && \
    source /opt/conda/bin/activate o3d_ml && \
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y && \
    pip install open3d==0.18.0 laspy tensorboard tqdm"

# Create point_transformer environment
RUN bash -c "source /opt/conda/bin/activate && \
    conda create -n point_transformer python=3.7 -y && \
    source /opt/conda/bin/activate point_transformer && \
    conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y && \
    conda install -c anaconda h5py pyyaml -y && \
    conda install -c conda-forge sharedarray tensorboardx -y && \
    pip install typing-extensions numpy laspy tqdm"

WORKDIR /app
COPY . /app

# Compile/install parts that depend on local code in the point_transformer environment
RUN bash -c "source /opt/conda/bin/activate point_transformer && \
    cd /app/point_transformer/lib/pointops && \
    python setup.py install"

# Set default environment to hugging_face and expose Gradio port
ENV PATH /opt/conda/envs/hugging_face/bin:$PATH
EXPOSE 7860

CMD ["python", "point_cloud_viewer.py"]
