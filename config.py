import configargparse
from utils import str2bool

def get_parser():
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", is_config_file=True, help="config文件路径")
    parser.add_argument("--seed", type=int, default=20, help="随机种子")

    parser.add_argument('--num_workers', type=int, default=0)

    # 定义此次训练的session，这个主要是方便后续三个session并行训练
    parser.add_argument('--session', type=int, default=1, help="定义此次训练的session")
    parser.add_argument("--num_of_class", type=int, default=3, help="定义类别数量, 对于不同数据集类别数不同")

    # 定义训练相应的参数
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument("--log_interval", type=int, default=1)

    # 定义优化器的参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 定义学习率变化的参数
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=False)

    # 定义迁移学习相关
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss_type', type=str, default='dann')

    # 定义存储路径
    parser.add_argument("--path", type=str, default = "E:\\EEG_DataSets\\SEED\\ExtractedFeatures\\")
    parser.add_argument("--tmp_saved_path", type=str,  default="E:\\EEG\\logs\\default\\")

    # 定义PRPL参数
    parser.add_argument("--model_type", type=str, default = "base", help="选择的模型")
    parser.add_argument("--cluster_weight", type=int, default = 2)
    parser.add_argument("--lower_rank", type=int, default = 32)
    parser.add_argument("--upper_threshold", type=float, default=0.9)
    parser.add_argument("--lower_threshold", type=float, default=0.5)

    # 定义是否存储模型
    parser.add_argument('--saved_model', type=str2bool, default=False, help="当前训练过程是否存储模型")

    parser.add_argument('--start_by', type=int, default=7)
    parser.add_argument('--t_epochs', type=int, default=14)

    return parser