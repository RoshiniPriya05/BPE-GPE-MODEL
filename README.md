# BPE vs GPE Model Comparison

This project implements and compares Byte Pair Encoding (BPE) and Gaussian Process Embedding (GPE) models for text processing and representation learning.

## Project Structure

```
├── models/
│   ├── bpe_model.py      # BPE implementation
│   ├── gpe_model.py      # GPE implementation
│   └── base_model.py     # Base model interface
├── benchmark/
│   ├── benchmark.py      # Main benchmarking framework
│   ├── metrics.py        # Evaluation metrics
│   └── datasets.py       # Dataset utilities
├── utils/
│   ├── preprocessing.py  # Text preprocessing utilities
│   ├── visualization.py  # Plotting and visualization
│   └── data_loader.py    # Data loading utilities
├── experiments/
│   ├── run_benchmark.py  # Main experiment runner
│   └── config.py         # Configuration settings
├── data/                 # Data directory
├── results/              # Benchmark results
└── requirements.txt      # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Benchmark

```bash
python experiments/run_benchmark.py
```


### Running BPE and Grapheme (Char) Model Benchmark

First, ensure you have installed the requirements:

```bash
pip install -r requirements.txt
```

To train and test the BPE model:

```bash
python train_bpe.py --model_type bpe --vocab_size 200 --test_text "என் பெயர் ரோஷினி பிரியா."
```

To train and test the grapheme (character-level) model:

```bash
python train_bpe.py --model_type char --test_text "என் பெயர் ரோஷினி பிரியா."
```

To benchmark both models and compare the number of tokens:

```bash
python train_bpe.py --model_type both --test_text "என் பெயர் ரோஷினி பிரியா."
```

The script will print the tokens, number of tokens, and a summary for both models. This helps you compare the efficiency of BPE and grapheme encoding for Tamil or any other language.

#### Required Files
- `data/tamil_corpus.txt` (your training corpus)
- `train_bpe.py` (the script)
- `requirements.txt` (should include `sentencepiece`)

#### Results
Model files will be saved in the `models/` directory. You can change the test sentence with the `--test_text` argument.

## Features

- **BPE Implementation**: Complete Byte Pair Encoding with vocabulary building and encoding/decoding
- **GPE Implementation**: Gaussian Process Embedding with kernel functions and optimization
- **Comprehensive Benchmarking**: Multiple metrics and datasets for comparison
- **Visualization**: Interactive plots and analysis tools
- **Modular Design**: Easy to extend and customize

## Benchmarks

The benchmark compares models on:
- Vocabulary coverage
- Encoding efficiency
- Semantic similarity preservation
- Training time
- Memory usage
- Reconstruction quality

## Results

Results are saved in the `results/` directory with detailed analysis and visualizations. 