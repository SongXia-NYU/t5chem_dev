import argparse
import os


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data_dir", type=str, required=True, help="root data dir")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_type", type=str, required=True,
                        help="Task type to use. ('product', 'reactants', 'reagents', 'regression', 'classification', 'pretrain', 'mixed')")
    parser.add_argument("--pretrained_model", default='',
                        help="Path to a pretrained model. If not given, we will train from scratch")
    parser.add_argument("--vocab", default='', help="Vocabulary file to load.")
    parser.add_argument("--tokenizer", default='', help="Tokenizer to use. ('simple', 'atom', 'selfies', 'pl')")
    parser.add_argument("--random_seed", default=8570, type=int, help="The random seed for model initialization")
    parser.add_argument("--num_epoch", default=100, type=int, help="Number of epochs for training.")
    parser.add_argument("--log_step", default=5000, type=int, help="Logging after every log_step")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and validation.")
    parser.add_argument("--init_lr", default=5e-4, type=float, help="The initial leanring rate for model training")
    parser.add_argument("--num_classes", type=int,
                        help="The number of classes in classification task. Only used when task_type is Classification")
    parser.add_argument("--vocab_name", default="simple")
    parser.add_argument("--add_tokens", action="append")
    parser.add_argument("--type_path", default="train")
    parser.add_argument("--comment",
                        help="Yes, this argument is required. Take some notes here so you can remember what you were doing 2 months later.")
    parser.add_argument("--split_file", default=None, help="relative path to data_dir")
    # yes, dtype is str. It will be converted during runtime
    parser.add_argument("--val_size", default=None, type=str, help="Supports both int and float number")
    parser.add_argument("--legacy_data_handling", default=False, action="store_true",
                        help="Use legacy data handling. This is for backward compatibility. Please do not use this.")
    parser.add_argument("--split_frac", default = 0.8, type=float, help="The split you want to use for train/valid split, defaults to 0.8")

def smart_parse_args():
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--config_file")
    config_args, unk = config_parser.parse_known_args()

    t5_parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_args(t5_parser)
    if unk:
        args = t5_parser.parse_args()
    else:
        args = t5_parser.parse_args(["@" + config_args.config_file])
        args.config_file = config_args.config_file
    print(vars(args))
    return args


def solv_num_workers():
    try:
        n_cpu_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu_avail = None
    n_cpu = os.cpu_count()
    num_workers = n_cpu_avail if n_cpu_avail is not None else n_cpu
    return n_cpu_avail, n_cpu, num_workers


if __name__ == "__main__":
    smart_parse_args()
