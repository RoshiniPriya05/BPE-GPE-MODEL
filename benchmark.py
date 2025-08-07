import pickle
from collections import Counter
# from tokenizers import ByteLevelBPETokenizer  

import sentencepiece as spm
import regex as re

def load_gpe_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compute_bpe_metrics(model_path, corpus):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    token_counts = [len(sp.encode_as_pieces(line)) for line in corpus]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    vocab_size = sp.get_piece_size()
    return {"vocab_size": vocab_size, "avg_tokens": avg_tokens}

def get_gpe_merges(vocab):
    merges = set()
    for word_tuple in vocab:
        for token in word_tuple:
            if len(token) > 1:
                merges.add(token)
    return sorted(merges, key=len, reverse=True)

def get_graphemes(word):
    return re.findall(r'\X', word)

def gpe_encode(text, merges):
    graphemes = get_graphemes(text)
    tokens = graphemes[:]
    for merge in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] + tokens[i+1] == merge:
                tokens[i] = merge
                del tokens[i+1]
            else:
                i += 1
    return tokens

def compare_bpe_gpe_tokens(bpe_model_path, gpe_vocab_path, test_text):
    # BPE tokenization
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    bpe_tokens = sp.encode_as_pieces(test_text)
    print("BPE tokens:", bpe_tokens)
    print("Number of BPE tokens:", len(bpe_tokens))

    # GPE tokenization
    gpe_vocab = load_gpe_vocab(gpe_vocab_path)
    merges = get_gpe_merges(gpe_vocab)
    gpe_tokens = gpe_encode(test_text, merges)
    print("GPE tokens:", gpe_tokens)
    print("Number of GPE tokens:", len(gpe_tokens))

if __name__ == "__main__":
    # Example usage
    test_text = "என் பெயர் ரோஷினி பிரியா."
    bpe_model_path = "models/tamil_bpe.model"
    gpe_vocab_path = "models/gpe/vocab.pkl"
    compare_bpe_gpe_tokens(bpe_model_path, gpe_vocab_path, test_text)