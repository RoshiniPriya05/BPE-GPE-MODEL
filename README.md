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

### Individual Model Usage

```python
from models.bpe_model import BPEModel
from models.gpe_model import GPEModel

# BPE Model
bpe = BPEModel(vocab_size=1000)
bpe.train(text_data)
encoded = bpe.encode("sample text")

# GPE Model
gpe = GPEModel(embedding_dim=128)
gpe.train(text_data)
embeddings = gpe.get_embeddings("sample text")
```

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