# MPPI Control

This folder contains the MPPI control experiments used in the paper, including Hopper and Walker2d benchmarks with either real MuJoCo dynamics or learned world models.

All commands below assume you have already activated the project environment:

```bash
conda activate westworld
cd MPPI
```

## Structure

- `hopper/`: Hopper control, data collection.
- `walker2d/`: Walker2d control, data collection.
- `configs/`: MPPI config files for control and rollout collection

## Quick Start

### Real Dynamics MPPI

Run MPPI with the MuJoCo simulator:

```bash
python hopper/hopper_mppi_expert_refcost_world_model.py --config-name hopper_mppi
python walker2d/walker2d_mppi_expert_refcost_world_model.py --config-name walker2d_mppi
```

The main control configs are:

- `configs/hopper_mppi.yaml`
- `configs/walker2d_mppi.yaml`

### Learned World Model MPPI

To run MPPI with a learned world model:

1. Put the finetuned checkpoint in the corresponding model folder, for example:
   - `hopper/world_model_westworld/`
   - `walker2d/world_model_westworld/`
2. Edit the task config in `configs/hopper_mppi.yaml` or `configs/walker2d_mppi.yaml`.
3. Set:
   - `use_world_model: true`
   - `wm_type`: one of `westworld`, `trajworld`, `tdm`, `mlpensemble`
   - `wm_ckpt`: checkpoint path
   - `wm_cfg`: model YAML path
4. Run:

```bash
python hopper/hopper_mppi_expert_refcost_world_model.py --config-name hopper_mppi
python walker2d/walker2d_mppi_expert_refcost_world_model.py --config-name walker2d_mppi
```

Example:

```yaml
use_world_model: true
wm_type: westworld
wm_ckpt: hopper/world_model_westworld/westworld_hopper.ckpt
wm_cfg: hopper/world_model_westworld/WestWorld.yaml
```

## Collect MPPI Rollouts

Use these scripts to collect MPPI rollout data:

```bash
python hopper/hopper_mppi_collect_pt.py --config-name hopper_mppi_collect
python walker2d/walker2d_mppi_collect_pt.py --config-name walker2d_mppi_collect
```

The collection configs are:

- `configs/hopper_mppi_collect.yaml`
- `configs/walker2d_mppi_collect.yaml`

Generated rollouts are saved under `output/` by default.

## Preprocess Rollouts

### Hopper

Inspect raw rollouts:

```bash
python hopper/unitraj_hoppermppi_tools.py inspect-hoppermppi \
  --datasets-root ./ \
  --env mppi_rollouts_hopper \
  --scan-minmax \
  --max-episodes 10
```

Convert raw rollouts to episode chunks:

```bash
python hopper/unitraj_hoppermppi_tools.py to-pt-hoppermppi \
  --datasets-root ./ \
  --env mppi_rollouts_hopper \
  --out ./hoppermppi_pt \
  --chunk-size 100000
```

Normalize converted episodes:

```bash
python hopper/unitraj_hoppermppi_tools.py pt-normalize \
  --pt-root ./hoppermppi_pt \
  --env mppi_rollouts_hopper \
  --clip-after-norm \
  --overwrite
```

## Notes

- The MPPI code targets legacy Gym and `mujoco-py`.
- On headless servers, these environment variables may be useful:

```bash
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy
```
