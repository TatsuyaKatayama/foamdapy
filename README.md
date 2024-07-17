![GitHub Repo stars](https://img.shields.io/github/stars/TatsuyaKatayama/foamdapy)
[![Python](https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/OpenFOAM-v2212%20%7C%20v2312-blue)](https://www.openfoam.com/news/main-news/openfoam-v2312)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TatsuyaKatayama/foamdapy/blob/develop/LICENSE)


# foamdapy
## Overview
This is a Python library of data assimilation with OpenFOAM.

## Requirement
* Python => 3.10.0
* OpenFOAM => v2212 (useable PyFoam)
* poetry 

## Install
if OpenFOAM is not installed, 
```bash
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get update
sudo apt-get install openfoam2312-default
echo 'source /usr/lib/openfoam/openfoam2312/etc/bashrc' >> ~/.bash_profile
source ~/.bash_profile
```

if you dont have Python => 3.10.0,
```bash
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile                    echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile
sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
pyenv install 3.10.13
pyenv global 3.10.13
```

if you don't have poetry,
```bash
curl -sSL https://install.python-poetry.org | python -
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bash_profile
source ~/.bash_profile
```

Now that the environment is ready, let's install it.
```bash
git clone https://github.com/TatsuyaKatayama/foamdapy.git
cd foamdapy/
poetry config virtualenvs.in-project true --local
poetry install
```


## Usage
Ensembel Forcast with OpenFOAM and data assimilation with LETKF,
Setup required for unensembled calculations. See the notebook for details.

```python
import random
from foamdapy import EnSim
random.seed(0)

n_cells = 3312
x_names = ["U.air", "U.oil","alpha.air","alpha.oil", "p_rgh"]
n_x_scaler = 9
y_names = ["U.air", "U.oil","alpha.air","alpha.oil", "p_rgh"]
n_y_scaler = 9 
n_ensemble = 20
num_cpus = 12

n_obs_cells = n_cells//2
obs_cells = random.sample(range(n_cells), n_obs_cells)
obs_cells.sort()

ensim = EnSim("path/to/ensemble_case_dir", "ensemble_case_prefix_", x_names, n_x_scaler, n_cells, n_ensemble, y_names,n_y_scaler, obs_cells, "obs_case",num_cpus)

#Forcast
ensim.ensemble_forcast("0.1")

#Observation. (true + noize in this case.)
ensim.observation("0.1")
y_noise = np.array([np.random.normal(0,np.sqrt(sc)) for sc in ensim.R_diag])
ensim.y0 += y_noise

#data assimilation
ensim.letkf_update()
```

## Features
One time data assimilation with LETKF(localize ensembel transform Kalman Filter).

```python
import numpy as np
from scipy.sparse import coo_matrix
from foamdapy.tools import letkf_update
from foamdapy.tools import createRdiag_from_xf

np.random.seed(0)
num_ensenble = 40
num_obs = 2

xf = np.stack(
    [
        np.random.normal(0.0, 1, num_ensenble),
        np.random.normal(1.0, 2, num_ensenble),
        np.random.normal(2.0, 1, num_ensenble),
    ],
    axis=1,
)

Hcoo = coo_matrix(([1, 1], ([0, 1], [1, 2])))
Hlil = Hcoo.tolil()
t0 = xf * 2.0
y0 = t0[:, -num_obs:].mean(axis=0)
y_indexes = np.array([1, 2])
R_diag = createRdiag_from_xf(xf, 3, y_indexes)
lmat = np.full((3, 3), 1.0)

num_cpu = 1
xa = letkf_update(xf, Hlil, y0, R_diag, y_indexes, lmat, num_cpu)
```


## Licence
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TatsuyaKatayama/foamdapy/blob/develop/LICENSE)

