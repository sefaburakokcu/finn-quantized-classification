# Low-Precision Neural Networks for Classification 

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Train](#train)
- [Evaluate](#evaluate)
- [Export](#export)
- [References](#references)


## Introduction

This repo contains training, evaluation and deploy scripts of some models for classification.
For training, [Brevitas](https://github.com/Xilinx/brevitas) which is a PyTorch research library for quantization-aware training (QAT) is utilized and [FINN](https://github.com/Xilinx/finn) which is an experimental framework from Xilinx Research Labs to explore deep neural network inference on FPGAs is used for deploying models on a [PYNQ-Z2](http://www.pynq.io/board.html) board.

* Available Models
  * Fully Connected(FC)
  * Convolutional Neural Network(CNV)
  * Lenet5(using HardTanh instead of Tanh activation)(LENET5)
  * Tiny MobilenetV1(TINYIMAGENET)

* Available Datasets
  * MNIST
  * CIFAR10
  * TINYIMAGENET


## Requirements

* Python >= 3.6.
* Pytorch >= 1.5.0.
* Brevitas >= 0.5.0.


## Usage
```
usage: quantized_classification_train.py [-h] [--datadir DATADIR]
                                         [--experiments EXPERIMENTS]
                                         [--dry_run] [--log_freq LOG_FREQ]
                                         [--evaluate] [--resume RESUME]
                                         [--detect_nan | --no_detect_nan]
                                         [--num_workers NUM_WORKERS]
                                         [--gpus GPUS]
                                         [--batch_size BATCH_SIZE] [--lr LR]
                                         [--optim OPTIM] [--loss LOSS]
                                         [--scheduler SCHEDULER]
                                         [--milestones MILESTONES]
                                         [--momentum MOMENTUM]
                                         [--weight_decay WEIGHT_DECAY]
                                         [--epochs EPOCHS]
                                         [--random_seed RANDOM_SEED]
                                         [--network NETWORK] [--pretrained]
                                         [--onnx_export]

PyTorch MNIST/CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  --datadir DATADIR     Dataset location
  --experiments EXPERIMENTS
                        Path to experiments folder
  --dry_run             Disable output files generation
  --log_freq LOG_FREQ
  --evaluate            evaluate model on validation set
  --resume RESUME       Resume from checkpoint. Overrides --pretrained flag.
  --detect_nan
  --no_detect_nan
  --num_workers NUM_WORKERS
                        Number of workers
  --gpus GPUS           Comma separated GPUs
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               Learning rate
  --optim OPTIM         Optimizer to use
  --loss LOSS           Loss function to use
  --scheduler SCHEDULER
                        LR Scheduler
  --milestones MILESTONES
                        Scheduler milestones
  --momentum MOMENTUM   Momentum
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --epochs EPOCHS       Number of epochs
  --random_seed RANDOM_SEED
                        Random seed
  --network NETWORK     neural network
  --pretrained          Load pretrained model
  --onnx_export         Export final model as ONNX

```

## Train

To start training a model from scratch, e.g. LFC_1W1A, run:
 ```bash
BREVITAS_JIT=1 python quantized_classification_train.py --network LFC_1W1A --experiments /path/to/experiments/
 ```

## Evaluate

To evaluate a model, e.g. LFC_1W1A, run:
 ```bash
BREVITAS_JIT=1 python quantized_classification_train.py --evaluate --network LFC_1W1A --resume /path/to/checkpoint.tar
 ```
 
## Export

To export a model, e.g. LFC_1W1A, run:
 ```bash
BREVITAS_JIT=1 python quantized_classification_train.py --evaluate --network LFC_1W1A --resume /path/to/checkpoint.tar --onnx_export
 ```

## References
- [BNN-PYNQ](https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq)
- [nn_benchmark](https://github.com/QDucasse/nn_benchmark)
