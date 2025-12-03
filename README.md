#  BFE-HAA-Net

## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build BFE-HAA-Net with:

```
git clone https://github.com/03100076/BFE-HAA-Net
cd BFE-HAA-Net
python setup.py build develop

```
## Quick Start

### Train

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
bash train_net.sh
```
### Test
To evaluate the model after training, run:

```
bash test_net.sh
```
### Download Checkpoint
- **Download Link**: [https://pan.baidu.com/s/16_tBVv03p6gs9VXYF5GJVQ](https://pan.baidu.com/s/16_tBVv03p6gs9VXYF5GJVQ)
- **Extraction Code**: `y9cv`
- **Files Included**:
  - `BFE-HAA-Net.yaml` - Model configuration file
  - Pre-trained model weights
