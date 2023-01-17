import torch

from mol_tokenizers import SimpleTokenizer
from transformers import T5ForConditionalGeneration
from t5chem.model import T5ForProperty

def demo():
    tokenizer = SimpleTokenizer(vocab_file="t5chem/vocab/simple.pt")
    tokenizer.create_vocab(vocab_file="t5chem/vocab/simple.pt")
    model_input = tokenizer("Fill-Mask:CC(C)[C@@H](C(=O)O)N", return_tensors="pt", return_token_type_ids=False)
    print(model_input["input_ids"].shape)
    model = T5ForProperty.from_pretrained("trained_models/models/USPTO_500_MT")
    model_out = model(**model_input)
    print(model_out.encoder_last_hidden_state.shape)


class HiddenStateExtractor:
    def __init__(self, vocab_file: str, pretrained_model_path: str) -> None:
        self.vocab_file = vocab_file
        self.pretrained_model_path = pretrained_model_path

        self._tokenizer = None
        self._model = None

    def extract_smi(self, smi: str, add_fill_mask=False) -> torch.Tensor:
        """
        Extract hidden embeddings from a SMILES sequence.
        Output is a torch tensor of size (1, N_seq, 256), where 1 is the batch size, N_seq is the total number
        of tokens after the tokenization of the SMILES and 256 is the hidden dimension
        """
        if add_fill_mask:
            smi = "Fill-Mask:" + smi
        model_input = self.tokenizer(smi, return_tensors="pt", return_token_type_ids=False)
        model_out = self.model(**model_input)
        hidden_state: torch.Tensor = model_out.encoder_last_hidden_state
        return hidden_state

    @property
    def model(self):
        if self._model is None:
            model = T5ForProperty.from_pretrained(self.pretrained_model_path)
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tokenizer = SimpleTokenizer(vocab_file=self.vocab_file)
            tokenizer.create_vocab(vocab_file=self.vocab_file)
            self._tokenizer = tokenizer
        return self._tokenizer

def demo_extractor():
    extractor = HiddenStateExtractor(vocab_file="t5chem/vocab/simple.pt", 
        pretrained_model_path="trained_models/models/USPTO_500_MT")
    emb1 = extractor.extract_smi("Fill-Mask:CC(C)[C@@H](C(=O)O)N", add_fill_mask=False)
    print(emb1.shape)
    emb2 = extractor.extract_smi("CC(C)[C@@H](C(=O)O)N", add_fill_mask=True)
    print(emb1.shape)
    # the two embeddings should be exactly the same
    print((emb1-emb2).abs().sum())


if __name__ == "__main__":
    demo()
    print("-"*20)
    demo_extractor()
