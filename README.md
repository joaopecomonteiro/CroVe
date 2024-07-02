# CroVe
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the official PyTorch implementation of CroVe

## Data generation
To train the single camera model you will need to:
1. Download the OPV2V dataset which can be found at https://mobility-lab.seas.ucla.edu/opv2v/.
2. Cd to `CroVe`
3. Edit the `configs/datasets/opv2v.yml` file, setting the `dataroot` and `label_root` entries to the location of the OPV2V dataset and the desired ground truth folder respectively.
4. Run the data generation script: `python scripts/make_opv2v_labels.py`. Bewarned there's a lot of data so this will take a few hours to run! 

## Training

Single camera:
Once ground truth labels have been generated, you can train our method by running the `train.py` script in the root directory: 
```
python train.py
```
Single camera is the default training setting.

Intermidiate fusion:
You can train the collaborative model using intermediate fusion by running the `train.py` script in the root directory: 
```
python train.py --fusion intermediate
```

