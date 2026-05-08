python walker2d/unitraj_walker2dmppi_tools.py to-pt-walker2dmppi \
  --datasets-path output/walker2d/expert_dist_init_base_action_expert_collect/expert_rollouts \
  --env mppi_rollouts_walker2d \
  --out output/walker2d/expert_dist_init_base_action_expert_collect/expert_rollouts/extract_top_5 \
  --sort-by cost --select-ratio 0.05 --chunk-size 1000000
  