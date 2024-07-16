import torch

class Tokenizer:
    def __init__(self):
        self.vocab = {'<Start>': 0}
        self.reverse_vocab = {0: '<Start>'}

    def _create_reverse(self):
        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k
        
    def fit(self, sentences: list[str]):
        for sentence in sentences:
            for c in sentence:
                if c not in self.vocab:
                    self.vocab[c] = len(self.vocab)
        
        self._create_reverse()
    
    def encode(self, sentence: str):
        return torch.tensor([0] + [self.vocab[c] for c in sentence] + [1])

    def encode_batch(self, sentences: list[str]):
        return [self.encode(sentence) for sentence in sentences]
    
    def decode(self, tokens: list[int]):
        return ''.join([self.reverse_vocab[t] for t in tokens])
    
