# Point Cloud Classification Tool

## 1. Introduction

The Point Cloud Classification Tool enables efficient classification of point cloud data, supporting .las and .laz file formats. Deployed within a Docker environment, it features a Gradio-based graphical user interface (GUI) for intuitive operation and visualization. Users can select between Point Transformer or RandLA-Net for classification. To optimize processing, the tool down-samples the point cloud to 1-meter resolution before classification. The final classification results retain the original format and resolution, ensuring improved accuracy.

---

## 2. Usage

### 2.1 Data Input

- Input format: **.las** or **.laz** files.

### 2.2 Tool Execution

The tool is deployed in a Docker environment. Users must first install Docker. It can be set up either by building from source or by pulling a pre-built Docker image from Docker Hub, enabling direct execution.

#### 2.2.1 Prerequisites

- **Install Docker:** [Docker Installation Guide](https://docs.docker.com/engine/install/)
- **Install NVIDIA Container Toolkit:** [NVIDIA Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### 2.2.2 Running via Prebuilt Docker Image

To run the tool using a pre-built image:

```sh
# Download the Docker image
docker pull gdaosu/usgs_lidar_classification:v1.0

# Run the tool
docker run --gpus 'all,"capabilities=compute,utility,graphics"' -p 7860:7860 -it gdaosu/usgs_lidar_classification:v1.0
```

##### **Visualization**

- Open a browser and enter:

    ```
    <hostname>:7860
    ```

- Replace `<hostname>` with the machine’s hostname, which can be obtained by running:

    ```
    hostname
    ```

#### 2.2.3 Building the Docker Image Locally

To build and run the Docker image from the source:

```
# Navigate to the source directory
cd /<path_to_point_cloud_visualizer_folder>

# Build the Docker image
docker build -t pcd_classification .

# Run the tool
docker run --gpus 'all,"capabilities=compute,utility,graphics"' -p 7860:7860 -it pcd_classification
```

##### Visualization

- Open a browser and enter:

    ```
    <hostname>:7860
    ```

- Replace `<hostname>` with the actual hostname obtained using:

    ```
    hostname
    ```

------

### 2.3 Output

- The final classification results retain the original format and resolution.

------

### 2.4 Troubleshooting

#### Issue: Unsupported GPU

##### Error Message:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```