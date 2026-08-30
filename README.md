# DeepImageWatermarking

## Environment Setup

### Prerequisites

- Git;
- 64-bit Windows or Linux;
- Anaconda, Miniconda, or Miniforge with the `conda` command available;
- an NVIDIA GPU with a compatible driver, only if GPU acceleration is required.

### Create the Environment

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate deep-image-watermarking
```

On Windows, use Anaconda Prompt or a Command Prompt in which Conda is
initialized. If the environment already exists and `environment.yml` has
changed, update it with:

```bash
conda env update -f environment.yml --prune
```

### Verify the Installation

```bash
python -c "import torch, torchvision, numpy, skimage; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On a machine with a supported NVIDIA GPU, `torch.cuda.is_available()` should
return `True`. On a machine without an NVIDIA GPU, the same project runs on the
CPU.

## Authors

- David Toholj
- Luka Matić
