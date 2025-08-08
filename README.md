BPE-GPE Model Comparison Project
A Python-based NLP project comparing Byte Pair Encoding (BPE) and Grapheme-level (character-based) tokenization models on Tamil text. This tool benchmarks both models for vocabulary efficiency, token length, and reconstruction accuracy.

🚀 How to Run This Project (VS Code)
1. Clone the repository

git clone https://github.com/RoshiniPriya05/BPE-GPE-MODEL.git
cd BPE-GPE-MODEL
2. Create and activate a virtual environment

python -m venv venv
Activate the environment:

On Windows:


.\venv\Scripts\activate
On Mac/Linux:


source venv/bin/activate
3. Install dependencies

pip install -r requirements.txt
If requirements.txt is missing:


pip install numpy matplotlib
4. Run the Models and Benchmark
Train the BPE model:


python train_bpe.py
Train the Grapheme (Character-level) model:


python train_gpe.py
Run benchmarking to compare results:

python benchmark.py
📁 Project Structure

├── train_bpe.py           # Train and test the BPE model
├── train_gpe.py           # Train and test the Grapheme model
├── benchmark.py           # Compare and visualize both models
├── models/
│   ├── bpe_model.py
│   ├── gpe_model.py
│   └── base_model.py
├── utils/
│   ├── preprocessing.py
│   ├── visualization.py
│   └── data_loader.py
├── data/
│   └── tamil_corpus.txt   # Sample corpus used for training/testing
├── results/               # Benchmark results and plots
├── requirements.txt
└── README.md
🧪 Sample Input (from data/tamil_corpus.txt)

இந்த உலகம் அழகானது.
You can modify or expand the corpus in data/tamil_corpus.txt to test with different inputs.

📊 Output
Tokenized outputs for both models

Metrics like token count and reconstruction quality

Benchmark visualizations saved in the results/ folder
