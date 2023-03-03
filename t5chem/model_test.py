import torch
from data_utils import LineByLineTextDataset
from transformers import T5ForConditionalGeneration, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from mol_tokenizers import SimpleTokenizer,PLTokenizer

# TODO declare tokenizer
# TODO declare model
# TODO declare dataset
# TODO declare dataloader
manual = True
samples = "Fill-Mask:CC(C)C1=CC=C(C=C1)C2=CC=CC=C2"
pl = True
vocab_file = "vocab.pt"
model_path = None
model = T5ForConditionalGeneration.from_pretrained(model_path)
if pl:
    tokenizer = PLTokenizer(vocab_file=vocab_file)
else:
    tokenizer = SimpleTokenizer(vocab_file=vocab_file)
task_specific_params = {
    "Reaction": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 10,
        "num_return_sequences": 5,
        "decoder_start_token_id": tokenizer.pad_token_id,
    }
}
if pl:

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
if not manual:
    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=samples, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=1)
    for batch in dataloader:
        print(batch)
        break
else:
    inputs = tokenizer(samples, return_tensors="pt")
    breakpoint()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    inputs = collator(inputs)
    breakpoint()
    outputs = model.generate(**inputs, **task_specific_params['Reaction'])
    print(outputs)
    print(outputs.loss)
    print(outputs.logits.shape)
    print(outputs.logits)
    print(outputs.logits.argmax(dim=-1))
    print(tokenizer.decode(outputs.logits.argmax(dim=-1)[0]))








