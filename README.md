# Length-MAX Tokenizer

Length-MAX Tokenizer is a length-oriented, graph-partition based sub-word tokenizer that maximises average token length while scaling linearly to multi-terabyte corpora.
---

## ✨ Features

| Category | Highlight |
|----------|-----------|
| **Scalability** | Processes 1 TB corpus in **47 min** on 256 CPU cores (13.4 GB/s). |
| **Efficiency**  | 2.3 × single-core throughput vs. SentencePiece; 17–18 % GPU-memory reduction for LLMs. |
| **Quality**     | +12.8 % GLUE, +139 % MTEB Recall@100, GPT-4 prefers generated stories. |
| **Drop-in**     | Rust backend with Python bindings; HuggingFace `tokenizers` compatible JSON. |
| **Reproducible**| Dockerfile and `make benchmark` reproduce all tables/figures in the paper. |

---

## 🔧 Installation

```bash
# Rust toolchain (1.74+) & Python 3.8+
$ git clone https://github.com/your-org/lengthmax-tokenizer.git
$ cd lengthmax-tokenizer
$ pip install maturin  # or poetry / hatch
$ make install         # builds Rust backend & installs Python wheel
```

> Binary wheels for Linux/macOS (x86_64 + arm64) are published on PyPI:  
> `pip install lengthmax`.

---

## 🚀 Quick Start

### Train a 50 k vocabulary

```bash
lengthmax train --input ./data/*.txt \
               --vocab 50000 \
               --workers 32 \
               --output vocab_50k.json
```

### Encode / Decode

```python
from lengthmax import LengthMaxTokenizer

lm_tok = LengthMaxTokenizer.from_pretrained("vocab_50k.json")
ids = lm_tok.encode("How are you today?")
print(ids)
print(lm_tok.decode(ids))
```

### HuggingFace Integration

```python
from transformers import AutoTokenizer
hf_tok = AutoTokenizer.from_pretrained("./vocab_50k.json", trust_remote_code=True)
```

---

## 📊 Reproduce Paper Results

```bash
# 1. Download public corpus & pre-computed checkpoints
$ make data

# 2. Single-machine speed
$ make benchmark_cpu

# 3. Distributed throughput (needs mpirun or slurm)
$ make benchmark_cluster

# 4. GPT-2 finetune & downstream evaluation
$ make benchmark_gpt2
```

All raw logs, tables and figures will be saved under `./runs/`.

---

## 📜 Citation

If you use Length-MAX in your research, please cite:

```bibtex
@inproceedings{lengthmax2025,
  title     = {Length-MAX Tokenizer: Linear-Time Sub-word Tokenisation via Graph Partitioning},
  author    = {Your Name and Others},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## 🤝 Contributing

Bug reports, feature requests and pull-requests are welcome!  
Please open an issue first to discuss substantial changes.

### Development setup

```bash
$ git clone … && cd lengthmax-tokenizer
$ pip install -r requirements-dev.txt
$ pre-commit install
$ make test
```

---

## 📄 Licence

Length-MAX Tokenizer is released under the Apache 2.0 licence.
