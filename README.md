# BAShockSeg
Boundary-Aware Vision Mamba for Shock Wave Segmentation in Schlieren Images
## 🛠️ Environment Setup

### Prerequisites

- Python 3.10
- CUDA 11.6 (recommended for GPU acceleration)
- PyTorch 1.13.1

### Installation

```bash
# 1. Create and activate conda environment
conda create -n BAShockSeg python=3.10 -y
conda activate BAShockSeg

# 2. Install PyTorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV
pip install -U openmim
mim install mmcv-full

# 4. Install other dependencies
pip install mamba-ssm==1.2.0
pip install timm lmdb mmengine numpy
```

#### Run


``````shell
python main.py
``````

You can also use checkpoints for inference with the following command:

```shell
python test.py
```

```shell
python eval_compute.py
cd eval
python evaluate.py
```

## 🚀 Quick Start

> ⚠️ **Full code is currently under embargo and will be released after paper acceptance.** 
> 
> The installation instructions below are provided for preview purposes only.

## Contact

If you have any other questions, feel free to contact me at **lixue_1428@163.com**.
