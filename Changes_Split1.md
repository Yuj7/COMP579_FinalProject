# Changes From Original Environment

## 1) Environment API Changes

File: `drone_2d_custom_gym_env_package/drone_2d_custom_gym_env/drone_2d_env.py`

### Added constructor parameters

- `initial_force=25000`
- `initial_rotation_force=3000`

### Internal state additions

- `self.initial_force`
- `self.initial_rotation_force`

### Reset behavior update

- `reset()` now preserves and forwards the new force parameters when re-initializing.

### Initial movement update

- Random linear throw force changed from fixed range `random.uniform(0, 25000)` to `random.uniform(0, self.initial_force)`.
- Random rotational throw changed from fixed `random.uniform(-3000, 3000)` to `random.uniform(-self.initial_rotation_force, self.initial_rotation_force)`.

## 2) Environment Registration Defaults

File: `drone_2d_custom_gym_env_package/drone_2d_custom_gym_env/__init__.py`

- Added default registration kwargs:
  - `initial_force: 25000`
  - `initial_rotation_force: 3000`

## 3) Evaluation Script Fixes

File: `examples/eval.py`

### Model loading/path fixes

- Added `Path`-based model directory resolution:
  - `MODEL_DIR = Path(__file__).resolve().parent / "ppo_agents"`
- Load model with env attached directly:
  - `PPO.load(..., env=env)`
- Removed fragile `model.set_env(env)` pattern.

### Additional eval updates

- Added case selector for split-1 checkpoints:
  - `CASE_ID = 1` (switch to `2` for strong-force PPO checkpoint)
- Updated default model name to case-based checkpoint format:
  - `ppo_agent_case{CASE_ID}`
- Added explicit checkpoint existence validation before loading:
  - checks for `<model_path>.zip`
  - raises a clear error with guidance if the file is missing

## 4) New Training Script for PPO/SAC + Cases

File: `examples/train_model.py` (new)

Added:

- Single entry point that supports both algorithms:
  - `ALGO = "ppo"` or `ALGO = "sac"`
- Case selection:
  - `CASE_ID = 1` or `CASE_ID = 2`
- Case dictionary with configurable force ranges:
  - Case 1: weak force (`initial_force=5000`, `initial_rotation_force=600`)
  - Case 2: strong force (`initial_force=25000`, `initial_rotation_force=3000`)
- Model save naming convention:
  - `ppo_agent_case1`, `ppo_agent_case2`, `sac_agent_case1`, `sac_agent_case2`

## 5) New Benchmark Script for PPO/SAC Across Cases

File: `examples/benchmark_cases.py` (new)

Added:

- Loads both PPO and SAC checkpoints.
- Runs both algorithms on case 1 and case 2.
- Uses the same case config pattern as training script.
- Prints per-run reward summary.
