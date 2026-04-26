# Changes From Original Environment

## 1) Environment API Changes

File: `drone_2d_custom_gym_env_package/drone_2d_custom_gym_env/drone_2d_env.py`

### Added constructor parameters

- `initial_force=5000`
- `initial_rotation_force=600`
- `step_penalty=0.1`
- `goal_reward=100.0`
- `death_penalty=-100.0`
- `success_radius=25.0`

### Reset behavior update

- `reset()` now preserves and forwards the force parameters and the new reward/success parameters when re-initializing.

### Initial movement update

- Random linear throw force changed from fixed range `random.uniform(0, 25000)` to `random.uniform(0, self.initial_force)`.
- Random rotational throw changed from fixed `random.uniform(-3000, 3000)` to `random.uniform(-self.initial_rotation_force, self.initial_rotation_force)`.
- Default stabilization delay was reduced from `n_fall_steps=10` to `n_fall_steps=5`.

### Reward logic redesign

- Removed the old dense distance-based reward:
  - `1 / (|dx| + 0.1) + 1 / (|dy| + 0.1)`
- Added per-step penalty reward structure:
  - each active step returns `-step_penalty`
  - success terminal step returns `goal_reward - step_penalty`
  - failure/timeout terminal step returns `death_penalty - step_penalty`
- Added explicit success detection through `reached_target(x, y)`.
- Success is not just touching the target. The drone must:
  - be within `success_radius` of the target
  - not already be in a failure state
  - be stable enough, meaning low x/y velocity, low angular velocity, and small tilt
- Added terminal status reporting in `info["terminal_status"]`:
  - `"success"`
  - `"failure"`
  - `"timeout"`

## 2) Environment Registration Defaults

File: `drone_2d_custom_gym_env_package/drone_2d_custom_gym_env/__init__.py`

- Added default registration kwargs:
  - `n_fall_steps: 5`
  - `initial_force: 5000`
  - `initial_rotation_force: 600`
  - `step_penalty: 0.1`
  - `goal_reward: 100.0`
  - `death_penalty: -100.0`
  - `success_radius: 25.0`

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
  - `CASE_ID = 1`
- Updated default model name to case-based checkpoint format:
  - `ppo_agent_case{CASE_ID}`
- Added explicit checkpoint existence validation before loading:
  - checks for `<model_path>.zip`
  - raises a clear error with guidance if the file is missing
- Updated default evaluation environment settings to the easier profile:
  - `n_fall_steps=5`
  - `initial_force=5000`
  - `initial_rotation_force=600`

## 4) New Training Script for PPO/SAC + Cases

File: `examples/train_model.py` (new)

Added:

- Single entry point that supports both algorithms:
  - `ALGO = "ppo"` or `ALGO = "sac"`

- Case selection:
  - `CASE_ID = 1` or `CASE_ID = 2`

- Case dictionary with configurable force ranges:
  - Case 1: weak force (`initial_force=5000`, `initial_rotation_force=600`)
  - Case 2: medium-hard force (`initial_force=12000`, `initial_rotation_force=1500`)

- Model save naming convention:
  - `ppo_agent_case1`, `ppo_agent_case2`, `sac_agent_case1`, `sac_agent_case2`

## 5) New Benchmark Script for PPO/SAC Across Cases

File: `examples/benchmark_cases.py` (new)

Added:

- Loads both PPO and SAC checkpoints.
- Runs both algorithms on case 1 and case 2.
- Uses the same case config pattern as training script.
- Prints per-run reward summary
