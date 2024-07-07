This is our anonymous repository for PRCV2024.

## Environment Preparation

#### 1. Clone CLIP into your folder
```sh
pip install git+https://github.com/openai/CLIP.git
```

#### 2. Install other libraries using requirements.txt


## Data Preparation
To train all of our models, we extract videos into frames for fast reading. Please refer to [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) repo for the detailed guide of data processing.


## Model Zoo
Coming Soon

Input = #frame x #temporal clip x # spatial crop x size
#### Kinetics-400
| Architecture | #Input |  Top-1 Acc.(%) | config |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x1x1x224x224 | 82.0 | - |
| ViT-B/16 | 8x3x1x224x224 | 82.6 | - |
| ViT-L/14 | 8x1x1x224x224 | 85.8 | - |
| ViT-L/14 | 8x3x1x224x224 | 86.6 | - |

#### HMDB-51
| Architecture | Task | #Input |  Top-1 Acc.(%) | config |
|:------------:|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | All | 8x1x1x224x224 | 74.6 | - |
| ViT-B/16 | 2-shot | 8x1x1x224x224 | 62.7 | - |
| ViT-B/16 | zero-shot | 8x1x1x224x224 | 45.8 | - |



## Training
```sh
# For Kinetics-400, use 8 frames and ViT-B/16.
bash scripts/run_train.sh  configs/k400/k400_train_rgb_vitb-16-f8.yaml

# For HMDB-51, use 8 frames and ViT-B/16. 
bash scripts/run_train.sh  configs/k400/k400_train_rgb_vitb-16-f8.yaml
```

## Testing
```sh
bash script/run_test.sh <PATH_TO_CONFIG> <PATH_TO_MODEL>
```

## Acknowledgement
This repository is built based on [Text4Vis](https://github.com/whwu95/Text4Vis) and CLIP(https://github.com/openai/CLIP). Sincere thanks to their wonderful works.
