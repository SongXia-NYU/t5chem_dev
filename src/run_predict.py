import argparse
import os
from functools import partial

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration

from data import T5MolTokenizer, T5SelfiesTokenizer, TaskPrefixDataset, data_collator


def add_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The model path to be loaded.",
    )
    parser.add_argument(
        "--prediction",
        default='',
        type=str,
        help="The file name for prediction.",
    )
    parser.add_argument(
        "--task_prefix",
        default='Product:',
        help="Prefix of current task. ('Product:', 'Yield:', 'Fill-Mask:')",
    )
    parser.add_argument(
        "--tokenizer",
        default='smiles',
        help="Tokenizer to use. (Default: 'smiles'. 'selfies')",
    )
    parser.add_argument(
        "--max_source_length",
        default=300,
        type=int,
        help="The maximum source length after tokenization.",
    )
    parser.add_argument(
        "--max_target_length",
        default=100,
        type=int,
        help="The maximum target length after tokenization.",
    )
    parser.add_argument(
        "--num_beams",
        default=5,
        type=int,
        help="Number of beams for beam search.",
    )
    parser.add_argument(
        "--num_preds",
        default=5,
        type=int,
        help="The number of independently computed returned sequences for each element in the batch.",
    )
    parser.add_argument(
        "--rep_penalty",
        default=1.0,
        type=float,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training and validation.",
    )


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.tokenizer == "smiles":
        Tokenizer = T5MolTokenizer
    else:
        Tokenizer = T5SelfiesTokenizer

    tokenizer = Tokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.pt'))

    if os.path.isfile(args.data_dir):
        args.data_dir, base = os.path.split(args.data_dir)
        base = base.split('.')[0]
    else:
        base = "test"

    testset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.task_prefix,
                                    max_source_length=args.max_source_length,
                                    max_target_length=args.max_target_length,
                                    type_path=base)
    data_collator_pad1 = partial(data_collator, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             collate_fn=data_collator_pad1)

    task_specific_params = {
        "Reaction": {
          "early_stopping": True,
          "max_length": args.max_target_length+1,
          "num_beams": args.num_beams,
          "num_return_sequences": args.num_preds,
          "decoder_start_token_id": tokenizer.pad_token_id,
          "repetition_penalty": args.rep_penalty,
        }
    }

    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    targets = []
    with open(os.path.join(args.data_dir, base+".target")) as rf:
        for line in rf:
            targets.append(line.strip()[:args.max_target_length])

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']

    predictions = [[] for i in range(args.num_preds)]
    if not args.prediction:
        args.prediction = os.path.join(args.model_dir, 'predictions.csv')
    
    for batch in tqdm(test_loader, desc="prediction"):

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        del batch['labels']
        with torch.no_grad():
            outputs = model.generate(**batch, **task_specific_params['Reaction'])
        for i,pred in enumerate(outputs):
            prod = tokenizer.decode(pred, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            predictions[i % args.num_preds].append(prod)

    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds

    test_df.to_csv(args.prediction, index=False)

if __name__ == "__main__":
    main()
