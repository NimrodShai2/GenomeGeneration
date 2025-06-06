# 🧬 Genome VAE Generator

A Variational Autoencoder (VAE)-based generative model for synthesizing novel microbial genomes from real genomic sequences in FASTA format.

---

## 🧪 Biological Motivation

In microbial genomics, species exhibit genomic diversity due to rapid evolution, mutations, horizontal gene transfer, and structural variations. Traditional genomic databases often undersample this diversity.

This project aims to:

- Model intra-species genomic diversity through deep generative modeling.
- Generate novel, biologically plausible genome sequences.
- Provide synthetic genomes for simulating genomic evolution, validating bioinformatics pipelines, and augmenting reference genome collections.

---

## 🧠 Technology Stack

- **Python** for scripting.
- **Biopython** for FASTA processing and sequence manipulation.
- **PyTorch** for building and training the Variational Autoencoder.
- **k-mer Tokenization** to efficiently represent DNA sequences.

---

## 📂 Project Structure

```
project/
├── preprocess_fasta.py        # Process FASTA into tokenized sequences
├── train_vae.py               # Train the Variational Autoencoder
├── generate_vae.py            # Generate synthetic genome sequences
├── saved/
│   ├── model.pt               # Trained PyTorch model
│   ├── vocab.json             # k-mer vocabulary
│   └── config.json            # Model configurations
└── input.fasta                # User-provided input genomes
```

---

## 🚀 Quickstart Guide

### 📌 Step 1: Install Dependencies

```bash
pip install biopython torch numpy pandas tqdm matplotlib seaborn umap-learn scikit-learn
```

### 📌 Step 2: Prepare Your Input FASTA

Provide a FASTA file named `input.fasta` with multiple genomic sequences. Example:

```fasta
>genome1
ATGCGTAGCTAGCTACGATCG...
>genome2
CGATGCTAGCTAGCTGATCGA...
```

### 📌 Step 3: Preprocess and Train the VAE

```bash
python train_vae.py
```

This script:

- Extracts DNA chunks.
- Tokenizes sequences into k-mers.
- Trains the VAE model.
- Saves model artifacts to `saved/`.

### 📌 Step 4: Generate Synthetic Genomes

After training, generate sequences:

```bash
python generate_vae.py --num_samples 100 --output generated.fasta
```

This produces a new FASTA file (`generated.fasta`) containing synthetic genomic sequences.

---

## 🛠️ How the VAE Works

- **Encoder**: Compresses DNA sequences into a latent representation (mean `μ` and variance `σ²`).
- **Reparameterization Trick**: Samples latent vector `z` from the latent distribution in a differentiable manner:
  ```
  z = μ + σ × ε,  ε ~ N(0,1)
  ```
- **Decoder**: Decodes `z` into DNA sequences.
- **Loss Function**:
  - **Reconstruction loss** (Cross-entropy): Measures decoding accuracy.
  - **KL divergence**: Regularizes latent space distribution toward standard normal.

---

## 🔧 Configuration Parameters

Modify these directly in `train_vae.py`:

- `CHUNK_SIZE`: DNA chunk length (default: 1000)
- `STRIDE`: Step size between chunks (default: 500)
- `K`: k-mer length (default: 6)
- `LATENT_DIM`: Latent vector size (default: 32)
- `EMBED_DIM`: Embedding dimension for k-mers (default: 64)
- `HIDDEN_DIM`: Encoder/decoder hidden dimension (default: 128)
- `BATCH_SIZE`: Training batch size (default: 64)
- `EPOCHS`: Number of training epochs (default: 10)

---

## 📈 Evaluating Generated Genomes (Optional)

- **Mash or FastANI**: Compare synthetic and real genome distances.
- **Codon usage and GC content analysis**.
- **Visualize latent space**: Use t-SNE or UMAP.

---

## 📚 References

- Kingma & Welling (2013). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- Zou et al. (2020). [Microbiome Dynamics](https://www.nature.com/articles/s41576-020-00291-8).
- Sharma et al. (2018). [DNA2vec](https://academic.oup.com/bioinformatics/article/34/15/i68/5045761).

---

## 📝 License

This project is distributed under the MIT License.

---

## 🙌 Acknowledgments

- BV-BRC & UHGG databases for genomic data.
- PyTorch and Biopython communities for excellent tools.

