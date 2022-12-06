import logging
import os
import random
from typing import Dict

import numpy as np
import torch

from transformers import (Trainer, T5Config,
                          T5ForConditionalGeneration, TrainingArguments)

from data_utils import (AccuracyMetrics, CalMSELoss,
                        T5ChemTasks, TaskSettings)
from data_utils_v2 import get_dataset
from model import T5ForProperty
from mol_tokenizers import (AtomTokenizer, MolTokenizer, PLTokenizer, SelfiesTokenizer,
                            SimpleTokenizer)
from general_utils import smart_parse_args, solv_num_workers
from trainer import EarlyStopTrainer,T5ChemTrainer

tokenizer_map : Dict[str, MolTokenizer] = {
    'simple': SimpleTokenizer,  # type: ignore
    'atom': AtomTokenizer,  # type: ignore
    'selfies': SelfiesTokenizer,    # type: ignore
    'pl': PLTokenizer
}


def train(args):
    print(args)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(args.random_seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    assert args.task_type in T5ChemTasks, \
        "only {} are currenly supported, but got {}".\
            format(tuple(T5ChemTasks.keys()), args.task_type)
    task: TaskSettings = T5ChemTasks[args.task_type]

    if args.pretrained_model: # retrieve information from pretrained model
        if task.output_layer == 'seq2seq':
            model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model)
        else:
            model = T5ForProperty.from_pretrained(
                args.pretrained_model, 
                head_type = task.output_layer,
            )
        if not args.tokenizer:
            if not hasattr(model.config, 'tokenizer'):
                logging.warning("No tokenizer type detected, will use SimpleTokenizer as default")
            tokenizer_type = getattr(model.config, "tokenizer", 'simple')
        else:
            tokenizer_type = args.tokenizer
        vocab_path = os.path.join(args.pretrained_model, 'vocab.pt')
        if not os.path.isfile(vocab_path):
            vocab_path = args.vocab
            if not vocab_path:
                raise ValueError(
                        "Can't find a vocabulary file at path '{}'.".format(args.pretrained_model)
                    )
        tokenizer = tokenizer_map[tokenizer_type](vocab_file=vocab_path)
        tokenizer.create_vocab()

        model.config.tokenizer = tokenizer_type # type: ignore
        model.config.task_type = args.task_type # type: ignore
    else:
        if not args.tokenizer:
            warn_msg = "This model is trained from scratch, but no \
                tokenizer type is specified, will use simple tokenizer \
                as default for this training."
            logging.warning(warn_msg)
            args.tokenizer = 'simple'
        assert args.tokenizer in tokenizer_map.keys(), "{} tokenizer is not supported."
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/'+args.vocab_name+'.pt')
        tokenizer = tokenizer_map[args.tokenizer](vocab_file=vocab_path)
        tokenizer.create_vocab()
        config = T5Config(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_past=True,
            num_layers=4,
            num_heads=8,
            d_model=256,
            tokenizer=args.tokenizer,
            task_type=args.task_type,
        )
        if task.output_layer == 'seq2seq':
            model = T5ForConditionalGeneration(config)
        else:
            model = T5ForProperty(config, head_type=task.output_layer, num_classes=args.num_classes)
    
    if args.tokenizer == "pl":
        print("Using PLTokenizer")
        added_tokens = ["<mod>", "</mod>"]
        aa_tokens = ["A", "G", "I", "L", "M", "P", "V", "F", "W", "N",
                        "C", "Q", "S", "T", "Y", "D", "E", "R", "H", "K"]
        # two extra capping AAs, B for ACE and J for NME
        capping_aa_tokens = ["B", "J"]
        added_tokens.extend(["<PROT>"+aa for aa in aa_tokens])
        added_tokens.extend(["<PROT>"+aa for aa in capping_aa_tokens])
        assert len(set(added_tokens)) == len(added_tokens), added_tokens
        tokenizer.add_tokens(added_tokens)
        model.resize_token_embeddings(len(tokenizer))

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))
    print("getting datasets")
    train_dataset,eval_dataset,eval_strategy, data_collator_padded,split = get_dataset(tokenizer,task,args)

    if task.output_layer == 'regression':
        compute_metrics = CalMSELoss
    elif args.task_type == 'pretrain':
        compute_metrics = None  
        # We don't want any extra metrics for faster pretraining
    else:
        compute_metrics = AccuracyMetrics

    n_cpu_avail, n_cpu, num_workers = solv_num_workers()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy=eval_strategy,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.log_step,
        per_device_eval_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=1,
        learning_rate=args.init_lr,
        prediction_loss_only=(compute_metrics is None),
        dataloader_num_workers=num_workers
    )

    trainer = T5ChemTrainer(
        t5chem_args = args,
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    print(tokenizer.get_vocab())
    trainer.train()
    print(args)
    print("logging dir: {}".format(training_args.logging_dir))
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    print("entering")
    args = smart_parse_args()
    print("args parsed")
    train(args)
