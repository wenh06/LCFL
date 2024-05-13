# Novel clustered federated learning based on local loss (LCFL)

## Introduction

This is the code for the paper [Novel clustered federated learning based on local loss] (to add link).

The code is based on a simulation framework for federated learning [fl-sim](https://github.com/wenh06/fl-sim).

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Usage

One can use the code in a Jupyter notebook or in command line as follows:

```bash
fl-sim configs/lcfl-rot-mnist.yml
```

or run in background using:

```bash
mkdir -p .logs
nohup fl-sim configs/lcfl-rot-mnist.yml > .logs/lcfl-rot-mnist.log 2>&1 & echo $! > .logs/lcfl-rot-mnist.pid
```

## Inspection of the results

One can use the `Panel` widget to inspect the results in a Jupyter notebook:

```python
%load_ext autoreload
%autoreload 2

from fl_sim.utils.viz import Panel

panel = Panel()
```

## Results

**ALL** experiments were repeated 5 times with different random seeds.

### Experiments on the RotatedMNIST dataset

 1200 clients - ACC           |  1200 clients - loss
:----------------------------:|:-----------------------------:
![RotatedMNIST-1200-acc](images/RotatedMNIST-1200-val-acc.svg) | ![RotatedMNIST-1200-loss](images/RotatedMNIST-1200-val-loss.svg)
 2400 clients - ACC           |  2400 clients - loss
![RotatedMNIST-2400-acc](images/RotatedMNIST-2400-val-acc.svg) | ![RotatedMNIST-2400-loss](images/RotatedMNIST-2400-val-loss.svg)

### Experiments on the RotatedCIFAR10 dataset

 dynamic transforms - ACC     | dynamic transforms - loss
:----------------------------:|:-----------------------------:
![RotatedCIFAR10-dt-acc](images/RotatedCIFAR10-dynamic-transform-val-acc.svg) | ![RotatedCIFAR10-dt-loss](images/RotatedCIFAR10-dynamic-transform-val-loss.svg)
 no transforms - ACC          |  no transforms - loss
![RotatedMNIST-nt-acc](images/RotatedCIFAR10-no-transform-val-acc.svg) | ![RotatedMNIST-nt-loss](images/RotatedCIFAR10-no-transform-val-loss.svg)

### v.s. other norms

FEMNIST                |  RotatedMNIST
:---------------------:|:---------------------:
![FEMNIST](images/LCFLversusOtherNorm-FEMNIST-NEW.svg) | ![RotatedMNIST](images/LCFLversusOtherNorm-RotMNIST-NEW.svg)
