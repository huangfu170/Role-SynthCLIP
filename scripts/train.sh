export SWANLAB_MODE=offline
export WANDB_MODE=offline
accelerate launch --config_file scripts/acce_config.yaml train.py --train_csv_path data/traindataset/MLM_filter_qwenSharedGPT4V_filter_train_checked.csv --log_project_dir test_multi_positive_base