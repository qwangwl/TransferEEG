import os

config_path = "configs\\transfer.yaml"
seed = 20
epochs = 1000
log_interval = 1
early_stop = 0
model_type = "base"

tmp_saved_path = "E:\\EEG\\logs\\1030\\base\\"
command = (
    f"python main.py "
    f"--seed {seed} "
    f"--config {config_path} "
    f"--n_epochs {epochs} "
    f"--log_interval {log_interval} "
    f"--model_type {model_type} "
    f"--tmp_saved_path {tmp_saved_path} "
    f"--early_stop {early_stop} "
    f"--transfer_loss_type dann "
)

os.system(command)

