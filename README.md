# SNOP
### Selective Nullification and Orthogonal Projection Unlearning

## Overview
SNOP is an unlearning algorithm that allows for the removal of specific data points from a model without retraining from scratch, while keeping the model's performance intact. SNOP is a gradient based method.

## Procedure To run the code
1. **CIFAR**: 
    - Navigate to the `CIFAR` directory.
    - Run the `finetune_resnet.py` script to fine-tune the ResNet model on CIFAR 100 data.
    - Use `SNOP.py` to perform SNOP unlearning on the CIFAR dataset.
    - Use the other scripts for specific unlearning methods: SSD, Gradient Ascent/Descent, Retain on Finetune, and retrain from scratch.

2. **VAE**:
    - Navigate to the `VAE` directory.
    - Run the `vae_utils.py` script to set up the VAE model.
    - Use `snop.py` to perform unlearning with the SNOP method on the saved VAE model.
    - Use the other scripts for specific unlearning methods: SSD, Gradient Ascent/Descent, Retain on Finetune, and retrain from scratch.


## Directory Structure
```
├── CIFAR
│   ├── SNOP.py
│   ├── SSD.py
│   ├── finetune_resnet.py
│   ├── finetune_retain_cifar.py
│   ├── grad_ascent_descent_cifar.py
│   ├── mia_cifar.py
│   ├── resnet.py
│   └── utils.py
│
└── VAE
    ├── snop.py
    ├── snop_vae_methodology.py
    ├── ft_retain_vae.py
    ├── grad_ascent_descent.py
    ├── pretrain_retain.py
    ├── ssd.py
    └── vae_utils.py
```

