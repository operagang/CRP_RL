# Learning-Based Container Retrieval

This repository provides a learning-based framework for the Container Retrieval Problem, including training code, benchmark evaluation, and comparisons with classical heuristic baselines.

---

## 1) Training with main.py

Training is executed via `main.py`.

Run:
- `python main.py`

What it does:
- Initializes the model/optimizer/logging via `initialize(args)`
- Runs training for `args.epochs` epochs using `train(...)`
- Saves logs/checkpoints using `save_log(...)`
- Evaluates the model on benchmark instances using `solve_benchmarks(...)` during training

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
