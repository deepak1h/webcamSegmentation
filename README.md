# Webcam Segmentation Application

This repository implements an image segmentation application using the YOLACT model. We leverage the power of the YOLACT model's yolact_base architecture for real-time instance segmentation. YOLACT is a state-of-the-art model known for its accuracy and efficiency in segmenting objects within images. With this application, you can perform live segmentation on images and video feeds, making it suitable for a wide range of computer vision tasks.

This repository contains code for a real-time webcam segmentation application. Follow the instructions below to set up the environment, install dependencies, and run the application.

## Prerequisites

- NVIDIA GPU with CUDA support (for GPU acceleration)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed
- Jupyter Notebook
- python  >3.8 <3.10
- Download [pretrained_model](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view) and put in /model directory

## Installation

1. Clone this repository:
   
```bash
   git clone https://github.com/deepak1h/webcamSegmentation.git
   cd webcamSegmentation
```
Create a Conda environment (replace webcam_segmentation_env with your preferred environment name):

```bash
    conda create -n webcam_segmentation_env python=3.8
```
Activate the Conda environment:

```bash
    conda activate webcam_segmentation_env
```
Install the required dependencies:

```bash
    pip install -r segmentation_requirements.txt
```

Setting Up CUDA
Ensure you have CUDA installed on your system for GPU acceleration. Refer to NVIDIA's CUDA Toolkit Installation [Guide]([https://www.openai.com/](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)) for instructions.
```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Usage
Running the Application Using Python
Activate the Conda environment if not already activated:

```bash
    conda activate webcam_segmentation_env
```
Run the application using Python:

```bash
    python webcamSegmentation.py
```
Press the 'q' key to exit the application.

Running the Application Using Jupyter Notebook
Activate the Conda environment if not already activated:

```bash
    conda activate webcam_segmentation_env
```
Start a Jupyter Notebook server:

bash
jupyter notebook
Open the webcamSegmentation.ipynb notebook.

Run the notebook cells to execute the application.

Acknowledgments
YOLACT for the instance segmentation model.
