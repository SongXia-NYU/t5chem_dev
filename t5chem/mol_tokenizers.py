import copy
import importlib
import itertools
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Set, Union

import torch
from torchtext.vocab import Vocab
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import logger, _is_end_of_word, _is_start_of_word
from transformers.tokenization_utils_base import TextInput, AddedToken

is_selfies_available: bool = False
if importlib.util.find_spec("selfies"):
    from selfies import split_selfies
    is_selfies_available = True
pattern: str = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex: re.Pattern = re.compile(pattern)
TASK_PREFIX: List[str] = ['Yield:', 'Product:', 'Fill-Mask:', 'Classification:', 'Reagents:', 'Reactants:']

class MolTokenizer(ABC, PreTrainedTokenizer):
    r"""
    An abstract class for all tokenizers. Other tokenizer should
    inherit this class
    """
    def __init__(
        self,
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        unk_token: str='<unk>',
        bos_token: str='<s>',
        pad_token: str="<pad>",
        eos_token: str='</s>',
        mask_token: str='<mask>',
        max_size: int=1000,
        task_prefixs: List[str]=[],
        **kwargs
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs)
        self.max_size = max_size
        task_prefixs = TASK_PREFIX+task_prefixs
        self.task_prefixs = copy.copy(task_prefixs)
        self.vocab_file = vocab_file
        self.source_files = source_files
        self.init_vocab_size = max_size-len(task_prefixs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def merge_vocabs(
        self, 
        vocabs: List[Vocab], 
        vocab_size: Optional[int]=None,
    ) -> Vocab:
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged: Counter = sum([vocab.freqs for vocab in vocabs], Counter())
        special_tokens: List[str] = list(self.special_tokens_map.values())  # type: ignore
        return Vocab(merged,
                    specials=special_tokens,
                    max_size=vocab_size-len(special_tokens) if vocab_size else vocab_size)

    def create_vocab(
        self, 
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        vocab_size: Optional[int]=None,
        ) -> None:
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
            vocab_size: (:obj:`int`, `optional`, defaults to `None`):
                The final vocabulary size. `None` for no limit.
        """
        if vocab_file is None:
            vocab_file = self.vocab_file
        if source_files is None:
            source_files = self.source_files
        if vocab_size is None:
            vocab_size = self.init_vocab_size

        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab: Vocab = self.merge_vocabs([torch.load(vocab_file)], vocab_size=vocab_size)

        elif source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter: Dict[int, Counter] = {}
            vocabs: Dict[int, Vocab] = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    num_lines = sum(1 for line in rf)
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file), total=num_lines):
                        try:
                            items: List[str] = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials: List[str] = list(self.special_tokens_map.values()) # type: ignore
                vocabs[i] = Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)
        else:
            self.vocab = None

        if self.vocab:
            extra_to_add: int = self.max_size - len(self.vocab)
            cur_added_len: int = len(self.task_prefixs) + 9 # placeholder for smiles tokens
            for i in range(cur_added_len, extra_to_add):
                self.task_prefixs.append('<extra_task_{}>'.format(str(i)))
            self.add_tokens(['<extra_token_'+str(i)+'>' for i in range(9)]+self.task_prefixs+['>'], special_tokens=True)
            self.unique_no_split_tokens = sorted(
                set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
            )

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize a molecule or reaction
        """
        pass

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        out_string: str = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, vocab_path: str) -> None:    # type: ignore
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)

class SimpleTokenizer(MolTokenizer):
    r"""
    Constructs a simple, character-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 100):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        breakpoint()
        return list(text)

class PLTokenizer(MolTokenizer):
    """
    A tokenizer that can tokenize both SMILES and Protein Sequences.
    Since there is overlap between protein sequence and SMILES, for example Serine (S) and Sulfer (S),
    we use the old reliable copypasta strategy to rewrite tokenize method
    """
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)
        self.unique_no_split_tokens.append("<mod>")
        self.unique_no_split_tokens.append("</mod>")
        self.PROT_PREFIX = "<PROT>"

    def _tokenize(self, text: str, is_prot: bool, **kwargs) -> List[str]:
        out = list(text)
        if is_prot:
            out = [self.PROT_PREFIX+s for s in out]
        return out

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Copypasta from Pretrained tokeinzer with some twist.
        Only the method: split_on_tokens is modified.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        )

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        # Try to avoid splitting on token
                        if (
                            i < len(split_text) - 1
                            and not _is_end_of_word(sub_text)
                            and not _is_start_of_word(split_text[i + 1])
                        ):
                            # Don't extract the special token
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result.append(full_word)
                            full_word = ""
                            continue
                    # Strip white spaces on the right
                    if tok_extended.rstrip and i > 0:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        sub_text = sub_text.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                    if i > 0:
                        sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            # This part of code is modified for PL tokenization #############
            out = []                                                        #
            is_prot = True                                                  #
            for token in tokenized_text:                                    #
                if token in self.unique_no_split_tokens:                    #
                    if token == "<mod>":                                    #
                        assert is_prot                                      #
                        is_prot = False                                     #
                    elif token == "</mod>":                                 #
                        assert not is_prot                                  #
                        is_prot = True                                      #
                    out.append(token)                                       #
                else:                                                       #
                    out.extend(self._tokenize(token, is_prot=is_prot))      #
                                                                            #
            return out                                                      #
            # End of code modification ######################################

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text
        

def recursive_tokenize(current_s: str, remaining_tokens: set, default_token_fn):
    """
    This is my unsuccessful attempt to write a custom tokenizer method.
    Safe to ignore.
    """
    if len(remaining_tokens) == 0:
        return default_token_fn(current_s)
    
    current_token = remaining_tokens.pop()
    out = []

    s_split = re.split(current_token, current_s)
    # print(s_split)
    if re.match(current_token, s_split[0]):
        out += [s_split[0]]
        s_split = s_split[1:]
    for i, ss in enumerate(s_split):
        if i%2 == 1:
            out += [ss]
        else:
            out += recursive_tokenize(ss, copy.copy(remaining_tokens), default_token_fn)
    return out


class AtomTokenizer(MolTokenizer):
    r"""
    Constructs an atom-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 1000):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        tokens: List[str] = [token for token in regex.findall(text)]
        assert text == ''.join(tokens), 'Error when parsing {}'.format(text)
        return tokens

class SelfiesTokenizer(MolTokenizer):
    r"""
    Constructs an SELFIES tokenizer. Based on SELFIES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 1000):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=1000, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)
        assert is_selfies_available, "You need to install selfies package to use SelfiesTokenizer"

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize a SELFIES molecule or reaction
        """
        return list(split_selfies(text))

