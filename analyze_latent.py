import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import umap
from vae_model import VAE
from preprocess_fasta import extract_chunks_from_fasta, build_kmer_vocab, tokenize_chunks


# -------------------------------
# GC Content Calculator
# -------------------------------
def gc_content(seq):
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / len(seq)


# -------------------------------
# Main Analysis Function
# -------------------------------
def analyze_latent(fasta_file, max_sequences=1000):
    os.makedirs("analysis", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and vocab
    with open("saved/config.json") as f:
        cfg = json.load(f)
    with open("saved/vocab.json") as f:
        vocab = json.load(f)

    model = VAE(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["seq_len"]
    ).to(device)
    model.load_state_dict(torch.load("saved/model.pt", map_location=device))
    model.eval()

    # Load real FASTA sequences (raw, not chunked) for length analysis
    raw_lengths = [len(r.seq) for r in SeqIO.parse(fasta_file, "fasta") if "N" not in r.seq]

    # Prepare fixed-length chunks for latent analysis
    print("Extracting and tokenizing fixed chunks...")
    chunks = extract_chunks_from_fasta(fasta_file, chunk_size=cfg["chunk_size"], stride=cfg["stride"])
    chunks = chunks[:max_sequences]
    tokenized = tokenize_chunks(chunks, cfg["k"], vocab)

    latents, gc_vals, lengths = [], [], []
    with torch.no_grad():
        for tokens, raw_seq in tqdm(zip(tokenized, chunks), total=len(chunks), desc="Encoding"):
            x = torch.tensor(tokens).unsqueeze(0).to(device)
            mu, _ = model.encode(x)
            latents.append(mu.squeeze().cpu().numpy())
            gc_vals.append(gc_content(raw_seq))
            lengths.append(len(raw_seq))

    latents = np.array(latents)

    # -------------------------------
    # UMAP Visualization
    # -------------------------------
    print("Running UMAP...")
    reducer = umap.UMAP()
    projection = reducer.fit_transform(latents)

    # Save metadata as CSV
    df = pd.DataFrame({
        "gc": gc_vals,
        "length": lengths,
        "umap_1": projection[:, 0],
        "umap_2": projection[:, 1]
    })
    df.to_csv("analysis/metadata.csv", index=False)

    # -------------------------------
    # Plot UMAP
    # -------------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="umap_1", y="umap_2", hue="gc", palette="viridis", s=20)
    plt.title("Latent Space UMAP (colored by GC%)")
    plt.tight_layout()
    plt.savefig("analysis/latent_umap_gc.png")
    plt.legend()
    plt.close()

    # -------------------------------
    # GC Content Distribution
    # -------------------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(df["gc"], bins=40, kde=True)
    plt.title("GC Content of Fixed-Length Chunks")
    plt.xlabel("GC %")
    plt.tight_layout()
    plt.savefig("analysis/gc_distribution.png")
    plt.close()

    # -------------------------------
    # Length Distribution of Raw FASTA
    # -------------------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(raw_lengths, bins=40)
    plt.title("Original Genome/Contig Length Distribution")
    plt.xlabel("Length (bp)")
    plt.tight_layout()
    plt.savefig("analysis/original_fasta_lengths.png")
    plt.close()

    # -------------------------------
    # Summary Stats
    # -------------------------------
    stats = {
        "chunk_count": len(chunks),
        "mean_gc": float(np.mean(gc_vals)),
        "std_gc": float(np.std(gc_vals)),
        "chunk_length": cfg["chunk_size"],
        "raw_fasta_mean_length": float(np.mean(raw_lengths)),
        "raw_fasta_std_length": float(np.std(raw_lengths)),
        "raw_fasta_count": len(raw_lengths)
    }
    with open("analysis/stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("[✓] Latent + sequence analysis complete. See 'analysis/'.")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, default="input.fasta", help="Input FASTA file")
    parser.add_argument("--max", type=int, default=1000, help="Max number of fixed-length chunks to analyze")
    args = parser.parse_args()
    analyze_latent(args.fasta, max_sequences=args.max)
