# LongStandingRewardWrapper

Dense reward shaping wrapper for **MyoSuite** bipedal standing / balance tasks  
(e.g. `myoLegStandRandom`, `myoLegWalk-v0` with zero target velocity, etc.).

The wrapper adds continuous height, position, survival and action-regularisation signals on top of the original environment reward, and terminates episodes early when the agent falls.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [API Reference](#api-reference)
4. [Reward Decomposition](#reward-decomposition)
5. [Fall Detection](#fall-detection)
6. [Observation Space](#observation-space)
7. [Action Space](#action-space)
8. [Info Dictionary](#info-dictionary)
9. [Hyper-parameter Guide](#hyper-parameter-guide)
10. [Training Example (PPO)](#training-example-ppo)
11. [Known Issues & Tips](#known-issues--tips)
12. [License](#license)

---

## Quick Start

```python
from myosuite.utils import gym
from long_standing_reward_wrapper import LongStandingRewardWrapper  # or paste the class

env = gym.make(
    "myoLegStandRandom",
    max_episode_steps=1000,
    reset_type="init",
    target_x_vel=0.0,
    target_y_vel=0.0,
)

env = LongStandingRewardWrapper(
    env,
    episode_len=1000,
    position_weight=2.0,
    height_weight=1.5,
    survival_weight=0.02,
    fall_penalty=20.0,
    success_bonus=25.0,
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

---

## Installation

```bash
pip install myosuite gymnasium stable-baselines3 mujoco
# optional but recommended
pip install tensorboard rich
```

The wrapper only needs:

```python
from myosuite.utils import gym   # MyoSuite’s gym/gymnasium shim
import numpy as np
```

---

## API Reference

### Constructor

```python
LongStandingRewardWrapper(
    env,
    *,
    episode_len: int = 1000,
    position_weight: float = 2.0,
    height_weight: float = 1.5,
    upright_weight: float = 1.2,      # currently unused (see notes)
    survival_weight: float = 0.02,
    action_weight: float = 0.002,
    fall_penalty: float = 20.0,
    success_bonus: float = 25.0,
    min_height: float = 0.70,
    min_uprightness: float = 0.70,    # currently unused
    max_xy_error: float = 0.20,
    max_z_error: float = 0.20,
    height_scale: float = 0.05,
    xy_scale: float = 0.07,
    z_scale: float = 0.10,
    use_action_penalty: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `episode_len` | `int` | `1000` | Maximum steps before forced truncation + success bonus |
| `position_weight` | `float` | `2.0` | Multiplier for the position (xy + z) Gaussian score |
| `height_weight` | `float` | `1.5` | Multiplier for the height-maintenance Gaussian score |
| `upright_weight` | `float` | `1.2` | *Reserved* – upright term is currently commented out |
| `survival_weight` | `float` | `0.02` | Constant reward given every non-terminal step |
| `action_weight` | `float` | `0.002` | Coefficient of mean-squared action penalty |
| `fall_penalty` | `float` | `20.0` | Subtracted from shaped reward when a fall is detected |
| `success_bonus` | `float` | `25.0` | Added when the agent survives the full `episode_len` |
| `min_height` | `float` | `0.70` | Pelvis height (m) below which the episode is terminated |
| `min_uprightness` | `float` | `0.70` | *Reserved* – upright fall check is currently disabled |
| `max_xy_error` | `float` | `0.20` | Max allowed horizontal distance from target (m) |
| `max_z_error` | `float` | `0.20` | Max allowed vertical distance from target (m) |
| `height_scale` | `float` | `0.05` | σ of the height Gaussian (metres) |
| `xy_scale` | `float` | `0.07` | σ of the xy-error Gaussian (metres) |
| `z_scale` | `float` | `0.10` | σ of the z-error Gaussian (metres) |
| `use_action_penalty` | `bool` | `True` | Whether to apply the action regularisation term |

---

## Reward Decomposition

The **total reward** returned by `step` is:

```text
reward = original_env_reward + shaped_reward
```

where

```text
shaped_reward =
      height_weight   * height_score
    + position_weight * position_score
    + survival_weight
    + action_penalty          # ≤ 0
    − fall_penalty            # only on fall
    + success_bonus           # only on full-episode survival
```

### 1. Height Score  (`height_score ∈ (0, 1]`)

```text
height_score = exp( − ((h − h₀) / height_scale)² )
```

- `h`  = current pelvis height (world Z of site/body `"pelvis"`)
- `h₀` = pelvis height recorded at `reset` (base height)
- `height_scale = 0.05 m` → score drops to ~0.37 when |h − h₀| = 5 cm

```
1.0 |●
    |  ●
0.6 |    ●
0.2 |      ●●
0.0 |________●●●● → |Δh|
         0  5 10 cm
```

### 2. Position Score  (`position_score ∈ (0, 1]`)

```text
position_score = exp( − (xy_err / xy_scale)² − (z_err / z_scale)² )
```

- `xy_err = ‖pelvis_xy − target_xy‖₂`
- `z_err  = |pelvis_z − target_z|`
- Defaults: `xy_scale = 0.07 m`, `z_scale = 0.10 m`

A perfect match of pelvis to the target site yields `1.0`.  
Error of ~7 cm in xy or ~10 cm in z reduces the score to ~0.37.

### 3. Survival Reward

```text
survival_reward = survival_weight = 0.02   (constant every step)
```

Encourages the policy to stay alive as long as possible.

### 4. Action Penalty  (`≤ 0`)

```text
action_penalty = − action_weight * mean(action²)
```

- Typical muscle actions ∈ [0, 1] (after MyoSuite’s internal mapping)
- With `action_weight = 0.002` the penalty is very mild (≈ −0.001 for average activation 0.7)

### 5. Fall Penalty

```text
shaped_reward −= fall_penalty     # default 20.0
terminated = True
```

Applied the instant any fall condition is met (see below).

### 6. Success Bonus

```text
if step_count >= episode_len and not fallen:
    shaped_reward += success_bonus   # default 25.0
    truncated = True
```

### Typical magnitude (healthy standing)

| Term                | Approx. value per step |
|---------------------|------------------------|
| height × 1.5        | 1.3 – 1.5              |
| position × 2.0      | 1.6 – 2.0              |
| survival            | 0.02                   |
| action penalty      | −0.001                 |
| **shaped total**    | **≈ 3.0 – 3.5**        |
| fall event          | −20                    |
| full-episode bonus  | +25                    |

---

## Fall Detection

An episode is terminated (`terminated=True`) when **any** of the following holds:

| Condition | Default threshold | Reason string |
|-----------|-------------------|---------------|
| `height < min_height` | 0.70 m | `"low_height"` |
| `xy_error > max_xy_error` | 0.20 m | `"large_xy_error"` |
| `z_error > max_z_error` | 0.20 m | `"large_z_error"` |

> **Note**  
> The uprightness check is currently **commented out** because the pelvis body local-Z axis is not aligned with world-up in the MyoLeg model (see [Known Issues](#known-issues--tips)).

The reason is stored in `info["fall_reason"]`.

---

## Observation Space

The wrapper **does not alter** the observation.  
It simply forwards the observation returned by the underlying MyoSuite environment.

For `myoLegStandRandom` / `myoLegWalk` the observation is typically a flat `Box` containing:

| Block | Content (approximate) | Dim |
|-------|-----------------------|-----|
| Joint positions / velocities | qpos, qvel of the leg DOFs | ~40–60 |
| Muscle activations / lengths / forces | actuator state | ~80 |
| Root / pelvis pose | free-joint or site data | ~7–12 |
| Target information | velocity / position targets | variable |

Exact shape depends on the concrete environment ID and `obs_keys` configuration.  
Print it once:

```python
obs, _ = env.reset()
print(obs.shape, env.observation_space)
```

---

## Action Space

Again **unchanged** by the wrapper.

MyoSuite muscle-driven legs expose a continuous `Box` of shape `(n_muscles,)`  
(≈ 80 for the full MyoLeg).  

Actions are interpreted as normalised muscle excitations; the environment internally maps them to the actuator range (usually [0, 1]).

```python
print(env.action_space)          # Box(-1.0, 1.0, (80,), float32) or similar
print(env.action_space.sample().shape)
```

---

## Info Dictionary

After every `step` the following keys are added / updated:

| Key | Type | Description |
|-----|------|-------------|
| `fallen` | `bool` | Whether a fall was detected this step |
| `fall_reason` | `str` or `None` | `"low_height"`, `"large_xy_error"`, `"large_z_error"` |
| `survived_full_episode` | `bool` | `True` only on the success-bonus step |
| `balance_metrics` | `dict` | See below |
| `shaped_reward` | `float` | The pure shaping signal (before adding original reward) |
| `wrapped_reward` | `float` | Final reward returned to the agent |

### `info["balance_metrics"]`

```python
{
    "height":      float,   # pelvis world-Z
    "xy_error":    float,   # horizontal distance to target
    "z_error":     float,   # vertical distance to target
    "uprightness": float,   # 0–1 (currently unreliable – see notes)
    "step":        int,     # current step count inside the episode
}
```

Useful for logging, TensorBoard histograms, or custom callbacks.

---

## Hyper-parameter Guide

### Making standing easier (early training)

```python
LongStandingRewardWrapper(
    env,
    episode_len=500,          # shorter episodes
    min_height=0.55,          # more tolerant
    max_xy_error=0.35,
    max_z_error=0.30,
    fall_penalty=10.0,        # softer punishment
    survival_weight=0.05,     # stronger “stay alive” signal
)
```

### Making standing harder / more precise

```python
LongStandingRewardWrapper(
    env,
    episode_len=2000,
    min_height=0.80,
    max_xy_error=0.10,
    max_z_error=0.08,
    height_scale=0.03,        # sharper height peak
    xy_scale=0.04,
    position_weight=3.0,
    fall_penalty=30.0,
    success_bonus=50.0,
)
```

### Disabling action regularisation

```python
use_action_penalty=False
```

### Re-enabling uprightness (after fixing the axis)

1. Print the pelvis rotation matrix at a known good pose.
2. Choose the local axis whose world-Z component is closest to +1.
3. Uncomment the upright term in `_compute_reward` and the fall check in `_is_fallen`.

---

## Training Example (PPO)

The training script you supplied can be used almost as-is.  
A cleaned-up minimal version:

```python
from pathlib import Path
import numpy as np
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from long_standing_reward_wrapper import LongStandingRewardWrapper


def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(
            "myoLegStandRandom",
            max_episode_steps=1000,
            reset_type="init",
            target_x_vel=0.0,
            target_y_vel=0.0,
        )
        env = LongStandingRewardWrapper(env, episode_len=1000)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    N_ENVS = 16
    TOTAL_TIMESTEPS = 50_000_000
    SEED = 42
    LOG_DIR = Path("logs/ppo_standing")
    MODEL_DIR = Path("models/ppo_standing")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env, str(LOG_DIR / "monitor"))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.025,
        verbose=1,
        seed=SEED,
        device="auto",
        tensorboard_log=str(LOG_DIR),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 128])),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_stand",
    )

    eval_env = DummyVecEnv([make_env(100, SEED + 100)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    eval_env.obs_rms = vec_env.obs_rms   # share running stats

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=50_000 // N_ENVS,
        n_eval_episodes=25,
        deterministic=True,
    )

    model.learn(TOTAL_TIMESTEPS, callback=[checkpoint_cb, eval_cb], progress_bar=True)

    model.save(str(MODEL_DIR / "ppo_standing_final"))
    vec_env.save(str(MODEL_DIR / "vecnormalize.pkl"))
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
```

Launch with:

```bash
python train_standing.py
tensorboard --logdir logs/ppo_standing
```

---

## Known Issues & Tips

### 1. Uprightness ≈ 0.5 (or 0) at the start

The pelvis body frame in MyoLeg is **not** aligned with world-up.  
Local Z is roughly horizontal → the classic `1 − 2(x²+y²)` formula yields ~0.

**Work-around (already applied in the current code):**
- upright reward term is commented out
- upright fall check is commented out

To re-enable later:

```python
R = data.xmat[self._pelvis_body].reshape(3, 3)
print(R[2, :])          # which local axis is vertical?
up = R[:, 1]            # pick the correct column
uprightness = float(np.clip(up[2], 0.0, 1.0))
```

### 2. Missing sites

If `"pelvis"` or `"pelvis_target"` sites do not exist, the wrapper falls back to the pelvis **body** position and a zero target.  
Always verify once:

```python
print(env.unwrapped.mj_model.site("pelvis").id)          # should not raise
print(env.unwrapped.mj_model.site("pelvis_target").id)
```

### 3. Reward scale vs. original MyoSuite reward

The shaped terms (≈ 3 per step) dominate the sparse original reward.  
If you want pure shaping, replace instead of add:

```python
# inside step()
reward = float(shaped_reward)   # discard original
```

### 4. VecNormalize

When loading a trained model always restore the same `VecNormalize` statistics:

```python
vec_env = VecNormalize.load("models/ppo_standing/vecnormalize.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False
```

---

## License

This wrapper is provided as-is for research use with MyoSuite.  
MyoSuite itself is released under the Apache 2.0 license.

---

*Generated for LongStandingRewardWrapper – standing balance on MyoLeg / MyoSuite.*
