# Landslide mapping from Sentinel-2 imagery through change detection
Code repository for the paper "[Landslide mapping from Sentinel-2 imagery through change detection](https://arxiv.org/abs/2405.20161)" (accepted at IGARSS 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2405.20161-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.20161)


## Get started
To run code in this repository, you first have to install required libraries
```bash
pip install -r requirements.txt
```

Then, you have to download the S2 image dataset for each inventory by running the Jupyter Notebooks in `processing/`.

To train a model, you can run `train.py` specifying hyperparameters as arguments. Arguments can be given either as command line arguments (e.g. `python train.py --encoder="resnet50"`) or specified in a YAML configuration file (e.g. `python train.py --yaml="configs/bbunet.yaml"`), or even both. If both a YAML config and command line arguments are specified in the command, command line arguments eventually override YAML arguments. If an argument is not specified, a default value is used, as specified in `src/arg_parser.py`).

To specify a pretrained backbone through the `--pretrained_encoder_weights` argument, you have to manually download the corresponding "full checkpoint" from the [SSL4EO-S12 repository](https://github.com/zhu-xlab/SSL4EO-S12?tab=readme-ov-file#pre-trained-models) first.

In `configs/` you can find hyperparameters used to train the models reported in the paper. Thus, the experiments in the paper can be run simply by:
```bash
python train.py --yaml="configs/model.yaml"
```

## Checkpoints
The models trained in the paper can be found [at this link](https://drive.google.com/drive/folders/1351hEZeY2T67aGhD-ONNyLN8Cq0cAfzX?usp=sharing).
The following code is used to load a model:
```python
import torch
from argparse import Namespace
from src.model_utils import make_model

# load trained checkpoint
ckpt_path = "path/to/model.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')
hparams = ckpt['hparams']

# load model architecture
model = make_model(hparams)

# load trained weights into model
model.load_state_dict(ckpt['state_dict'])
```
