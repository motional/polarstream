## Installation
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Requirements

- Linux
- Python 3.7+
- PyTorch 1.6 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv) 

#### Notes

- A rule of thumb is that your pytorch cuda version must match the cuda version of your systsem for other cuda extensions to work properly. 

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.7.9
- PyTorch: 1.9
- CUDA: 11.2
- CUDNN: 8.1.0

### Basic Installation 

```bash
# basic python libraries
conda create --name polarstream python=3.7
conda activate polarstream
conda install pytorch torchvision cudatoolkit=11.2 -c pytorch
git clone https://github.com/qchenclaire/polarstream
cd polarstream
pip install -r requirements.txt

# add CenterPoint to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_POLARSTREAM"
```

### Advanced Installation

#### pytorch scatter
```bash
conda install pytorch-scatter -c pyg
```
#### nuScenes dev-kit

```bash
git clone https://github.com/tianweiy/nuscenes-devkit

# add the following line to ~/.bashrc and reactivate bash (remember to change the PATH_TO_NUSCENES_DEVKIT value)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_NUSCENES_DEVKIT/python-sdk"
```

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-11.2/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.2
export CUDA_HOME=/usr/local/cuda-11.2
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
# Note: nms build is gpu-specific. I built different nms under different environment for 1080ti and v100
bash setup.sh 
```

#### APEX
Note: apex build is also gpu-specific. 
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### spconv2.x
```bash
pip install spconv-cu114
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with all those pretrained models. 