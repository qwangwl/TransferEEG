import os

config_path = "configs\\transfer.yaml"
seed = 20
epochs = 1000
log_interval = 1
early_stop = 0

tmp_saved_path = f"E:\\EEG\\logs\\Base"
command = (
    f"python examples\\test_base.py "
    f"--seed {seed} "
    f"--dataset_name seed3 "
    f"--config {config_path} "
    f"--n_epochs {epochs} "
    f"--log_interval {log_interval} "
    f"--tmp_saved_path {tmp_saved_path} "
    f"--early_stop {early_stop} "
    f"--transfer_loss_type dann "
    f"--saved_model True "
)
os.system(command)

