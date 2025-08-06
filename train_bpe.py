def benchmark_models(input_file, test_text, vocab_size=200):
    """Train and compare BPE and grapheme models on the same test sentence."""
    results = {}
    for model_type in ["bpe", "char"]:
        model_prefix = f"models/tamil_{model_type}"
        train_sp_model(input_file, model_prefix, vocab_size, model_type)
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        pieces = sp.encode_as_pieces(test_text)
        results[model_type] = len(pieces)
        print(f"\nModel: {model_type.upper()}")
        print("Tokens:", pieces)
        print("Number of tokens:", len(pieces))
    print("\nSummary:")
    for model_type, num_tokens in results.items():
        print(f"{model_type.upper()} model: {num_tokens} tokens")

import sentencepiece as spm
import sys

def train_sp_model(input_file, model_prefix, vocab_size=200, model_type="bpe"):
    if model_type == "char":
        # For grapheme (character-level), vocab_size is ignored by SentencePiece
        spm.SentencePieceTrainer.Train(f"--input={input_file} --model_prefix={model_prefix} --model_type=char --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3")
    else:
        spm.SentencePieceTrainer.Train(f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3")

def encode_decode_example(model_prefix, text):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print("Original text:", text)
    pieces = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    print("Tokens:", pieces)
    print("Number of tokens:", len(pieces))
    print("Token IDs:", ids)
    decoded = sp.decode_ids(ids)
    print("Decoded text:", decoded)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and test BPE or Grapheme (char) model with SentencePiece.")
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'char', 'both'], help='Model type: bpe, char (grapheme), or both for benchmarking')
    parser.add_argument('--vocab_size', type=int, default=200, help='Vocabulary size (ignored for char model)')
    parser.add_argument('--test_text', type=str, default="என் பெயர் ரோஷினி பிரியா.", help='Text to encode/decode')
    args = parser.parse_args()

    input_file = "data/tamil_corpus.txt"
    if args.model_type == "both":
        benchmark_models(input_file, args.test_text, args.vocab_size)
    else:
        model_prefix = f"models/tamil_{args.model_type}"
        # Train the model
        train_sp_model(input_file, model_prefix, args.vocab_size, args.model_type)
        # Test encoding/decoding
        encode_decode_example(model_prefix, args.test_text)
