# Team Setup Guide

This guide shows how to set up the project on a new Windows machine so the training and evaluation scripts can run.

## 1) Install Miniconda

Install Miniconda from the official site:
https://docs.conda.io/en/latest/miniconda.html

After installation, open PowerShell.

## 2) Create and activate the environment

From the project root:

```powershell
conda create -n .venv python=3.9 -y
conda activate .venv
```

## 3) Install the local drone environment package

The project uses a local editable install so changes in the source code are picked up immediately.

```powershell
cd "..\Drone-2d-custom-gym-env-for-reinforcement-learning\drone_2d_custom_gym_env_package"
pip install -e .
pip install "numpy<2" "shimmy>=2.0"
```

## 5) Run training

Go to the examples folder:

```powershell
cd "..\Drone-2d-custom-gym-env-for-reinforcement-learning\examples"
```

Edit `train_model.py` and set:

- `ALGO = "ppo"` or `ALGO = "sac"`
- `CASE_ID = 1` or `CASE_ID = 2`

Then run:

```powershell
python train_model.py
```

This saves models into:

```text
examples\ppo_agents\
```

## 6) Run benchmark tests

After training checkpoints exist, run:

```powershell
python benchmark_cases.py
```

This compares:

- PPO case 1
- PPO case 2
- SAC case 1
- SAC case 2

## 7) Run the visual evaluator

To load and view a saved PPO model:

```powershell
python eval.py
```

## 8) Important notes

- Keep `numpy<2` in this environment because the project still uses the older Gym API.
- The project currently uses Gym-style environments with Stable-Baselines3 compatibility wrappers.
