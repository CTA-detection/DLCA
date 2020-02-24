# DLCA
![detection_visu](visu.gif)
#### left: input (3D volume)
#### right: output (3D volume. Blue boxes：predictions, red box：GT label）

### Table of contents

  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)


## Introduction

This repository contains the source code and the trained model for our work Deep Learning for Detecting Cerebral Aneurysms on CT Angiography.

## Prerequisites
- Ubuntu
- Python 3
- Pytorch 0.4.1
- NVIDIA GPU + CUDA CuDNN

This repository has been tested on NVIDIA TITAN Xp. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

## Installation
* Clone this repo:
```bash
git clone https://github.com/JiehuaYang/DLCA.git
cd DLCA
```
* Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Train 
* Run command as below.
```bash
python train.py -j=16 -b=12 --input="train_data/" --output="./checkpoint/"
```
### 2. Inference 
* Click the [checkpoint link(GoogleDrive)](https://drive.google.com/drive/folders/138_EpuZaMB0sS_dVmO0ux6_07sFfwRKZ?usp=sharing) to download trained model into "./checkpoint".
* Click the [data link(GoogleDrive)](https://drive.google.com/file/d/1M76tVZp-dqW9COlESnh0n8iuii5PKiS8/view?usp=sharing) to download test image "brain_CTA.nii.gz" 
* Run command as below.
```bash
# an example with the image named "brain_CTA.nii.gz"
python inference.py -j=1 -b=1 --resume="./checkpoint/trained_model.ckpt" --input="./test_image/brain_CTA" --output="./prediction/brain_CTA"
```


## Results

| Sensitivity | False Positive per case |
|:-------------:|:-------------:|
| 97.5% | 13.8| 

