# Learning-Based Container Retrieval

This repository provides the experimental datasets and code used in the following paper:

> **Woo-Jin Shin**, Inguk Choi, Sang-Hyun Cho, Hyun-Jung Kim.  
> *Learning to Retrieve Containers: A Scale-diverse Deep Reinforcement Learning Approach for the Container Retrieval Problem*.  
> *Transportation Research Part C: Emerging Technologies*, 2026.
> [https://doi.org/10.1016/j.trc.2025.105496](https://doi.org/10.1016/j.trc.2025.105496)

---

## 0) Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

(Optional) Install the exact environment used in our experiments:
```bash
pip install -r requirements-frozen.txt
```

---

## 1) Training with main.py

Training is executed via `main.py`.

Run:
- `python main.py`

All hyperparameters and settings are specified in `main.py` through `argparse.Namespace` (e.g., batch sizes, learning rate, instance size range, architecture options, online/offline setting).

---

## 2) baselines/

This folder contains pretrained final models, comparison algorithms, and testing code.

Pretrained models:
- `baselines/models/proposed/epoch(100).pt`
  - Trained model under the default (offline) setting
- `baselines/models/online/epoch(100).pt`
  - Trained model under the online setting

Comparison algorithms:
- `baselines/durasevic2025.py`
- `baselines/kim2016.py`
- `baselines/leveling.py`
- `baselines/lin2015.py`
- `baselines/lowerbound.py`

Testing script:
- `baselines/test.py`
  - Runs evaluation of trained models and baseline algorithms on benchmark instances

---

## 3) benchmarks/

This folder contains benchmark instances, parsing/evaluation utilities, and benchmark instance generation code.

Benchmark instances:
- `benchmarks/Lee_instances/`
  - Lee benchmark instances
- `benchmarks/Shin_instances/`
  - Instances generated in this paper

Utilities:
- `benchmarks/benchmarks.py`
  - Parsing benchmark instances and evaluating the model
- `benchmarks/generate_benchmarks.py`
  - Benchmark instance generation code

---

## 4) Environment

- `env/env.py`
  - Environment implementation (state, transitions, and reward computation)

---

## 5) Training Instance Generator

- `generator/generator.py`
  - Generates random training instances with varying layouts and container counts

---

## 6) Model Architecture (model/)

Neural network components:
- `model/decoder.py`
- `model/encoder.py`
- `model/model.py`
- `model/sampler.py`

---

## 7) Training Logic

- `trainer.py`
  - Core training loop, optimization logic, logging, and model initialization

---

© 2025 Woo-Jin Shin. The source code is released under the MIT License, and the dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See the LICENSE files for details.
