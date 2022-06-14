import unicodedata
import re
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def build_pairs(data_path):
    print("Reading lines...")
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')

    pairs = [[preparation(s) for s in l.split('\t')] for l in lines]
    pairs = [list(reversed(p)) for p in pairs]

    return pairs

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def preparation(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    
    return s

def prepare_dataset(data_path):
    pairs = build_pairs(data_path)
    pairs = filterPairs(pairs)
    eng = Vocab('eng')
    fra = Vocab('fra')
    for pair in pairs:
        eng.addSentence(pair[1])
        fra.addSentence(pair[0])
    return eng, fra, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH  and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(eng, fra, pair, device):
    input_tensor = tensorFromSentence(eng, pair[0], device)
    target_tensor = tensorFromSentence(fra, pair[1], device)
    return (input_tensor, target_tensor)
    
    

