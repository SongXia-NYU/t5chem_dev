import os
from functools import partial
from data_utils import (LineByLineTextDataset,TaskPrefixDataset,data_collator)

import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Subset, ConcatDataset, random_split
from sklearn.model_selection import train_test_split


def collect_files(suffix, data_dir):
    collected_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(suffix):
                collected_files.append(os.path.join(root, file))
    return collected_files

# Entry Point
# Here to keep t5chems legacy behavior.
def get_split_size(dataset, split):
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size
    return [train_set_size, valid_set_size]



def dataset_handling(tokenizer, task, args):
    if args.task_type == 'pretrain':
        files = collect_files(".txt", args.data_dir)
        datasets = []
        for data in files:
            datasets.append(LineByLineTextDataset(
                tokenizer=tokenizer,
                file_path=data,
                block_size=task.max_source_length,
                prefix=task.prefix,
            ))
        concat_dataset = ConcatDataset(datasets)
        do_eval = False
        # Do we split by eval? If so, we need to split the dataset
        # TODO determine if this should be an argument or if we check if val is in the files. Should we always split?
        if any("val" in s for s in list(map(os.path.basename,files))):
            train_dataset, eval_dataset = random_split(concat_dataset, get_split_size(concat_dataset,args.split_size))
            eval_strategy = "steps"
        else:
            train_dataset = concat_dataset
            eval_dataset = None
            eval_strategy = "no"
        data_collator_padded = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )


    else: #Classification, just copied from legacy for now
        train_dataset = TaskPrefixDataset(
            tokenizer,
            data_dir=args.data_dir,
            prefix=task.prefix,
            max_source_length=task.max_source_length,
            max_target_length=task.max_target_length,
            separate_vocab=(task.output_layer != 'seq2seq'),
            type_path=args.type_path,
        )
        data_collator_padded = partial(
            data_collator, pad_token_id=tokenizer.pad_token_id)

        do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
        if do_eval:
            eval_strategy = "steps"
            eval_dataset = TaskPrefixDataset(
                tokenizer,
                data_dir=args.data_dir,
                prefix=task.prefix,
                max_source_length=task.max_source_length,
                max_target_length=task.max_target_length,
                separate_vocab=(task.output_layer != 'seq2seq'),
                type_path="val",
            )
        else:
            eval_strategy = "no"
            eval_dataset = None
    split = None
    return train_dataset, eval_dataset, eval_strategy, data_collator_padded, split

def legacy_dataset_handling(tokenizer, task, args):
    if args.task_type == 'pretrain':
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=os.path.join(args.data_dir, 'train.txt'),
            block_size=task.max_source_length,
            prefix=task.prefix,
        )
        data_collator_padded = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    else:
        train_dataset = TaskPrefixDataset(
            tokenizer,
            data_dir=args.data_dir,
            prefix=task.prefix,
            max_source_length=task.max_source_length,
            max_target_length=task.max_target_length,
            separate_vocab=(task.output_layer != 'seq2seq'),
            type_path=args.type_path,
        )
        data_collator_padded = partial(
            data_collator, pad_token_id=tokenizer.pad_token_id)

    if args.task_type == 'pretrain':
        do_eval = os.path.exists(os.path.join(args.data_dir, 'val.txt'))
        if do_eval:
            eval_strategy = "steps"
            eval_dataset = LineByLineTextDataset(
                tokenizer=tokenizer,
                file_path=os.path.join(args.data_dir, 'val.txt'),
                block_size=task.max_source_length,
                prefix=task.prefix,
            )
        else:
            eval_strategy = "no"
            eval_dataset = None
    else:
        do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
        if do_eval:
            eval_strategy = "steps"
            eval_dataset = TaskPrefixDataset(
                tokenizer,
                data_dir=args.data_dir,
                prefix=task.prefix,
                max_source_length=task.max_source_length,
                max_target_length=task.max_target_length,
                separate_vocab=(task.output_layer != 'seq2seq'),
                type_path="val",
            )
        else:
            eval_strategy = "no"
            eval_dataset = None

    if args.split_file is not None:
        # use a split file instead of physically split datasets into training and evaluation set
        assert eval_strategy == "no"
        assert eval_dataset is None
        assert args.val_size is None

        split_f = os.path.join(args.data_dir, args.split_file)
        split = torch.load(split_f)
        eval_dataset = Subset(train_dataset, torch.as_tensor(val_index))  # TODO Where is this from
        train_dataset = Subset(train_dataset, torch.as_tensor(train_index))  # TODO Where is this from
        eval_strategy = "steps"

    if args.val_size is not None:
        # split during runtime
        assert args.split_file is None
        assert eval_dataset is None
        assert eval_strategy == "no"

        try:
            val_size = int(args.val_size)
        except ValueError:
            val_size = float(args.val_size)
        ds_size = len(train_dataset)
        #TODO switch this to torch random split.
        train_index, val_index = train_test_split(np.arange(ds_size), test_size=val_size, random_state=args.random_seed)
        eval_dataset = Subset(train_dataset, torch.as_tensor(val_index))
        train_dataset = Subset(train_dataset, torch.as_tensor(train_index))
        eval_strategy = "steps"
    split = None
    return train_dataset, eval_dataset, eval_strategy, data_collator_padded, split

def get_dataset(tokenizer, task, args):
    dataloading = legacy_dataset_handling if args.legacy_data_handling else dataset_handling
    print("Using legacy data handling: {}".format(args.legacy_data_handling))
    train_dataset, eval_dataset, eval_strategy, data_collator_padded, split = dataloading(tokenizer,task, args)
    print("dataloading complete")
    return train_dataset, eval_dataset, eval_strategy, data_collator_padded, split



