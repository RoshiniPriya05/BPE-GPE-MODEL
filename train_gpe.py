def get_gpe_merges(vocab):
    # Extract all merges from the learned vocab
    merges = set()
    for word_tuple in vocab:
        for token in word_tuple:
            if len(token) > 1:
                merges.add(token)
    # Sort merges by length descending (to apply longest merges first)
    return sorted(merges, key=len, reverse=True)

def gpe_encode(text, merges):
    # Split text into graphemes
    graphemes = get_graphemes(text)
    tokens = graphemes[:]
    # Iteratively apply merges
    for merge in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] + tokens[i+1] == merge:
                tokens[i] = merge
                del tokens[i+1]
            else:
                i += 1
    return tokens
import regex as re
from collections import Counter
import pickle

# Load corpus as a list of words
def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().split()

# Split a word into Unicode grapheme clusters
def get_graphemes(word):
    return re.findall(r'\X', word)

# Learn Grapheme Pair Encoding (GPE) merges
def learn_gpe(corpus, num_merges=1000):
    tokens = [tuple(get_graphemes(word)) + ('</w>',) for word in corpus]
    vocab = Counter(tokens)
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new_vocab = Counter()
        for word, freq in vocab.items():
            w = list(word)
            i = 0
            new_word = []
            while i < len(w):
                if i < len(w) - 1 and w[i] == best[0] and w[i+1] == best[1]:
                    new_word.append(w[i] + w[i+1])
                    i += 2
                else:
                    new_word.append(w[i])
                    i += 1
            new_vocab[tuple(new_word)] += freq
        vocab = new_vocab
    return vocab

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    corpus = load_corpus("data/tamil_corpus.txt")
    vocab = learn_gpe(corpus, num_merges=500)
    save_vocab(vocab, "models/gpe/vocab.pkl")
    print("GPE vocabulary trained and saved.")

    # Demonstrate GPE encoding for a sample sentence
    sample_text = "என் பெயர் ரோஷினி பிரியா."
    merges = get_gpe_merges(vocab)
    gpe_tokens = gpe_encode(sample_text, merges)
    print("Original text:", sample_text)
    print("GPE tokens:", gpe_tokens)
    print("Number of tokens:", len(gpe_tokens))
