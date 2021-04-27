# Denoise AutoEncoder For Tabular Data

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Documentation**](./docs/)
| [**Credit**](#credit)

[![Github License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

### Denoise AutoEncoder(DAE)
DAE is an [AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder#:~:text=An%20autoencoder%20is%20a%20type,to%20ignore%20signal%20%E2%80%9Cnoise%E2%80%9D.) model trained to perform denoise task. The model takes a partially corrupted input data and outputs the cleaned data.

Through the denoising task, the model learns the input distribution and produces latent representations that are robust to corruptions. The latent representations extracted from the model can be useful for a variety of downstream tasks. One can:  
    1. Use the latent representations to train supervised ML models, renders DAE as a vehicle for automatic feature engineering.  
    2. Use the latent representations for unsupervised tasks like similarity query or clustering.  

### Applying Denoise AutoEncoder to Tabular data  
To train DAE on tabular data, the most important piece is the noise generator. What makes sense and most effective is [swap noise](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629), through which, each value in the training data maybe replaced by a random value from the same column.

### What's included
This package implements:  
    1. Swap Noise generator.  
    2. Dataframe parser which converts arbitrary pandas dataframe to numpy arrays.  
    3. DAE network constructor with configurable body blocks.  
    4. DAE training function.  
    5. Sklearn style `.fit`, `.transform` API.  
    6. Sklearn style model also supports `save` and `load`. 

## Installation

tabular_dae is built with pyTorch. Make sure to install the dependencies listed in [requirements.txt](./requirements.txt). Then install the package using pip:
```bash
# download the requirements.txt file
pip install -r requirements.txt
pip install git+https://github.com/ryancheunggit/tabular_dae
```

## Quickstart

```python
import pandas as pd
from tabular_dae import DAE


# read data
df = pd.read_csv(<path-to-csv-file>)

# initialize a dae model
dae = DAE(
    body_network='deepstack',
    body_network_cfg=dict(hidden_size=1024),
    swap_noise_probas=.15,
    device='cuda',
)  

# fit the model
dae.fit(df, verbose=1, optimizer_params={'lr': 3e-4})

# extract latent representation with the model
latent = dae.transform(df)
```

## Credit  
```
@software{tabular_dae2021nielseniq,
  author = {Ren Zhang},
  title = {Denoise AutoEncoder for Tabular Data},
  url = {https://github.com/ryancheunggit/tabular_dae},
  version = {0.2},
  year = {2021},
}
```
