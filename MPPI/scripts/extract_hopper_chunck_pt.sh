python hopper/unitraj_hoppermppi_tools.py to-pt-hoppermppi \
  --datasets-path output/hopper/expert_dist_init_base_action_expert_collect/random_rollouts_6 \
  --env mppi_rollouts_hopper \
  --out output/hopper/expert_dist_init_base_action_expert_collect/random_rollouts_6/extract \
  --sort-by cost --select-ratio 0.1 --chunk-size 1000000