import os

config_path = "configs\\transfer.yaml"
seed = 20
epochs = 1000
log_interval = 1
early_stop = 0
model_type = "prpl"
num_of_class = 3

tmp_saved_path = f"E:\\EEG\\logs\\prpl"
command = (
    f"python examples\\test_prpl.py "
    f"--seed {seed} "
    f"--config {config_path} "
    f"--dataset_name seed3 "
    f"--n_epochs {epochs} "
    f"--log_interval {log_interval} "
    f"--tmp_saved_path {tmp_saved_path} "
    f"--early_stop {early_stop} "
    f"--transfer_loss_type dann "
    f"--num_of_class {num_of_class} "
    f"--session 1 "
)

os.system(command)

