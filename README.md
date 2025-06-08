# 🧬 Genome VAE Generator

A Variational Autoencoder (VAE)-based generative model for synthesizing novel microbial genomes from real genomic
sequences in FASTA format.

---

## 🧪 Biological Motivation

In microbial genomics, species exhibit genomic diversity due to rapid evolution, mutations, horizontal gene transfer and
structural variations. Traditional genomic databases often undersample this diversity.

This project aims to:

- Model intra-species genomic diversity through deep generative modeling
- Generate novel, biologically plausible genome sequences (this is still an open problem)
- Provide synthetic genomes for simulating genomic evolution, validating bioinformatics pipelines, and augmenting
  reference genome collections

---

## 🧠 Technology Stack

- Python for scripting
- Biopython for FASTA processing and sequence manipulation
- PyTorch for building and training the Variational Autoencoder
- k-mer tokenization to efficiently represent DNA sequences
- UMAP, Matplotlib and Seaborn for visualization of latent space and sequence statistics

---

## 📂 Project Structure

project_root/  
├── preprocess_fasta.py # Extract and tokenize DNA chunks  
├── vae_model.py # VAE class definition
├── model_factory.py # Factory for creating VAE models with different architectures
├── train_vae.py # Train the VAE model (CLI parameters)  
├── generate_vae.py # Generate synthetic genomes (samples length from input distribution)  
├── analyze_fasta.py # Analyze latent space and sequence stats for any FASTA  
├── saved/                     
│ ├── model.pt # Trained PyTorch model weights  
│ ├── vocab.json # k-mer vocabulary  
│ └── config.json # Model configuration  
├── analysis/ # Output directory for plots and stats  
└── input.fasta # User-provided input genomes
├── README.md # Project documentation
├── requirements.txt # Python dependencies

---

## 🚀 Quickstart Guide

### 1. Install Dependencies

    pip install biopython torch numpy pandas tqdm matplotlib seaborn umap-learn scikit-learn

### 2. Prepare Your Input FASTA

Provide a file named `input.fasta` containing multiple genomic sequences. Example:

    >genome1
    ATGCGTAGCTAGCTACGATCG...
    >genome2
    CGATGCTAGCTAGCTGATCGA...

### 3. Preprocess and Train the VAE

Run the training script with default parameters or override via CLI flags:

    python train_vae.py \
      --fasta_file input.fasta \
      --chunk_size 1000 \
      --stride 500 \
      --k 6 \
      --latent_dim 32 \
      --embed_dim 64 \
      --hidden_dim 128 \
      --batch_size 64 \
      --epochs 10 \
      --model_type vae \
      --lr 0.001 \
      --save_dir saved

**Training Script Arguments**

- `--fasta_file` (str, default=`input.fasta`): Path to multi-genome FASTA file
- `--chunk_size` (int, default=`1000`): Length of each DNA chunk (bp)
- `--stride` (int, default=`500`): Step size between chunks (bp)
- `--k` (int, default=`6`): k-mer size for tokenization
- `--latent_dim` (int, default=`32`): Dimensionality of the latent space
- `--embed_dim` (int, default=`64`): Dimension of k-mer embeddings
- `--hidden_dim` (int, default=`128`): Hidden layer size for encoder/decoder
- `--batch_size` (int, default=`64`): Number of samples per training batch
- `--epochs` (int, default=`10`): Number of training epochs
- `--lr` (float, default=`1e-3`): Learning rate for optimizer
- `--save_dir` (str, default=`saved`): Directory to save model weights and config
- `--model_type` (str, default=`vae`): Type of model to train (currently only `vae` or `conv` supported)

### 4. Generate Synthetic Genomes

Generate variable-length genomes by sampling real genome lengths:

    python generate_vae.py \
      --num_genomes 100 \
      --input_fasta input.fasta \
      --output generated.fasta

Each synthetic genome’s length is drawn at random from the lengths in `input.fasta`.

### 5. Analyze Sequences

Use the same analysis script for both real and generated FASTA files.

**Real (chunked) sequences**

    python analyze_fasta.py \
      --fasta input.fasta \
      --tag real \
      --mode chunked \
      --max 1000

**Generated (full) sequences**

    python analyze_fasta.py \
      --fasta generated.fasta \
      --tag generated \
      --mode full \
      --max 1000

Outputs in `analysis/`:

- `real_metadata.csv` / `generated_metadata.csv` — per-sequence GC%, length, UMAP coordinates
- `real_umap_gc.png` / `generated_umap_gc.png` — UMAP projection colored by GC%
- `real_gc_distribution.png` / `generated_gc_distribution.png` — GC% histogram
- `real_length_distribution.png` / `generated_length_distribution.png` — length histogram
- `real_stats.json` / `generated_stats.json` — summary statistics

---

## 🛠️ How the VAE Works

1. Encoder embeds k-mer tokens and compresses them into latent vectors (μ, σ²).
2. Reparameterization trick: sample z = μ + σ × ε (ε ∼ N(0,1)) in a differentiable way.
3. Decoder maps z back to k-mer sequences via a feedforward or convolutional network.
4. Loss combines cross-entropy reconstruction loss with KL divergence against N(0,I).

---

## 📈 Evaluating Generated Genomes

- Mash or FastANI for distance comparisons
- Codon usage and GC content analysis
- UMAP or t-SNE for latent space visualization

---

## 📚 References

- Kingma & Welling (2013) Auto-Encoding Variational Bayes
- Zou et al. (2020) Microbiome Dynamics
- Sharma et al. (2018) DNA2vec

---

## 📝 License

MIT License

---

## 🙌 Acknowledgments

- BV-BRC and UHGG for genomic data
- PyTorch and Biopython communities for their tools

Enjoy exploring microbial genome generation!  
