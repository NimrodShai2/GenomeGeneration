# 🧬 Genome VAE Generator

A Variational Autoencoder (VAE)-based generative model for synthesizing novel microbial genomes from real genomic sequences in FASTA format.

---

## 🧪 Biological Motivation

In microbial genomics, species exhibit genomic diversity due to rapid evolution, mutations, horizontal gene transfer and structural variations. Traditional genomic databases often undersample this diversity.

This project aims to:

- Model intra-species genomic diversity through deep generative modeling  
- Generate novel, biologically plausible genome sequences  
- Provide synthetic genomes for simulating genomic evolution, validating bioinformatics pipelines, and augmenting reference genome collections  

---

## 🧠 Technology Stack

- Python for scripting  
- Biopython for FASTA processing and sequence manipulation  
- PyTorch for building and training the Variational Autoencoder  
- k-mer tokenization to efficiently represent DNA sequences  
- UMAP, Matplotlib and Seaborn for visualization of latent space and sequence statistics  

---

## 📂 Project Structure

- project_root/  
    - preprocess_fasta.py        # Extract and tokenize DNA chunks  
    - train_vae.py               # Train the VAE model  
    - generate_vae.py            # Generate synthetic genomes (samples length from input distribution)  
    - analyze_fasta.py           # Analyze latent space and sequence stats for any FASTA  
    - saved/                     
        - model.pt               # Trained PyTorch model weights  
        - vocab.json             # k-mer vocabulary  
        - config.json            # Model configuration  
    - analysis/                  # Output directory for plots and stats  
    - input.fasta                # User-provided input genomes  

---

## 🚀 Quickstart Guide

### Step 1: Install Dependencies

    pip install biopython torch numpy pandas tqdm matplotlib seaborn umap-learn scikit-learn

### Step 2: Prepare Your Input FASTA

Provide a file named `input.fasta` containing multiple genomic sequences. Example:

    >genome1
    ATGCGTAGCTAGCTACGATCG...
    >genome2
    CGATGCTAGCTAGCTGATCGA...

### Step 3: Preprocess and Train the VAE

    python train_vae.py

This will:

- Extract fixed-length DNA chunks (e.g. 1000 bp windows)  
- Tokenize each chunk into k-mers  
- Train the VAE for a fixed number of epochs  
- Save artifacts in `saved/`  

### Step 4: Generate Synthetic Genomes

Now generate variable-length genomes by sampling real genome lengths:

    python generate_vae.py --num_genomes 100 --input_fasta input.fasta --output generated.fasta

Each synthetic genome will have its length drawn at random from the lengths in `input.fasta`.

---

## 📊 Analysis

Use the same script to analyze real or generated FASTA files.

### Analyze real (chunked) sequences

    python analyze_fasta.py --fasta input.fasta --tag real --mode chunked --max 1000

### Analyze generated (full) sequences

    python analyze_fasta.py --fasta generated.fasta --tag generated --mode full

Outputs in `analysis/`:

- `<tag>_metadata.csv` — per-sequence GC%, length, UMAP coordinates  
- `<tag>_umap_gc.png` — UMAP projection colored by GC%  
- `<tag>_gc_distribution.png` — GC% histogram  
- `<tag>_length_distribution.png` — length histogram  
- `<tag>_stats.json` — summary statistics  

---

## 🛠️ How the VAE Works

1. **Encoder**: Embeds k-mer tokens and compresses them into latent vectors (μ, σ²).  
2. **Reparameterization Trick**: Samples z = μ + σ × ε (ε ∼ N(0,1)) in a differentiable way.  
3. **Decoder**: Maps z back to k-mer sequences via a feedforward network.  
4. **Loss**: Combines cross-entropy reconstruction loss with KL divergence against N(0, I).

---

## 🔧 Configuration Parameters

Edit `train_vae.py` or override via command line:

- CHUNK_SIZE: DNA chunk length (default 1000)  
- STRIDE: Step between chunks (default 500)  
- K: k-mer length (default 6)  
- LATENT_DIM: Size of latent space (default 32)  
- EMBED_DIM: k-mer embedding size (default 64)  
- HIDDEN_DIM: Hidden layer size (default 128)  
- BATCH_SIZE: Training batch size (default 64)  
- EPOCHS: Number of training epochs (default 10)  

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
