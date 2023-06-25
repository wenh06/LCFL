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

## Results

to add images and tables....
