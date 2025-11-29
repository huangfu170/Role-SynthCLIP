cd /mnt/cpfs-data/scripts/train/clip_train
# head_node_ip=$MASTER_ADDR
# node_rank=$RANK
# echo $node_rank
export SWANLAB_MODE=offline
export WANDB_MODE=offline
# accelerate launch --config_file /mnt/cpfs-data/scripts/train/clip_train/acce_config.yaml --main_process_ip $head_node_ip --machine_rank $node_rank train.py --train_csv_path /mnt/cpfs-data/scripts/train/clip_train/data/MLM_filter_qwenSharedGPT4V_filter_train_checked.csv --log_project_dir MLM_filter_qwen_SharedGPT4V_filter_train_checked_large
accelerate launch --config_file /mnt/cpfs-data/scripts/train/clip_train/acce_config.yaml train.py --train_csv_path /mnt/cpfs-data/scripts/MLM_filter_qwen/SharedGPT4V_train_role_play_50_new_filter.csv --log_project_dir test_multi_positive_base