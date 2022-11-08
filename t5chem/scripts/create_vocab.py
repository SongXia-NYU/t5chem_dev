import copy
import re
from typing import List, Set
from mol_tokenizers import ModToekenizer, SimpleTokenizer


def create_vocab():
    tokenizer = ModToekenizer(None, source_files = "/scratch/sx801/scripts/t5chem_dev/data/smiles_train_0/train.txt", additional_special_tokens=("<mod>", "</mod>"))
    tokenizer.create_vocab()
    tokenizer.save_vocabulary("/scratch/sx801/scripts/t5chem_dev/t5chem/vocab/mol_tag_simple.pt")

def check_vocab():
    tokenizer = ModToekenizer(vocab_file="/scratch/sx801/scripts/t5chem_dev/t5chem/vocab/mol_tag_simple.pt", additional_special_tokens=("<mod>", "</mod>"))
    tokenizer = SimpleTokenizer(vocab_file="/scratch/sx801/scripts/t5chem_dev/t5chem/vocab/simple.pt")
    tokenizer.create_vocab()
    tokenizer.add_tokens(["<mod>", "</mod>"])
    print(tokenizer.vocab.stoi)
    print("*"*30)
    print(tokenizer.added_tokens_encoder)
    print("*"*30)
    # s = "<mod>C/C=C/c1cc(C(OCc2ccccc2)(C(F)(F)F)C(F)(F)F)ccc1N1CCNCC1C</mod>"
    s = "<mod>C/C=</mod>C/c1cc(C(</mod>OCc2c<mod>cccc2)(C(F)</mod>(F)F)C(F)(<mod>F)F)ccc1N1CCNCC1C</mod>"
    r = tokenizer.tokenize(s)
    print(r)
    print(tokenizer(s))

def check_rec():
    s = "<mod>C/C=</mod>C/c1cc(C(</mod>OCc2c<mod>cccc2)(C(F)</mod>(F)F)C(F)(<mod>F)F)ccc1N1CCNCC1C</mod>"
    r = recursive_tokenize(s, set([re.compile("(<mod>)"), re.compile("(<\/mod>)")]), lambda s: list(s))
    print(r)

if __name__ == "__main__":
    check_vocab()
