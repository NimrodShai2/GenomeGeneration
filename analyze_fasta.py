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
from preprocess_fasta import build_kmer_vocab, tokenize_chunks


def gc_content(seq):
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / len(seq)


def analyze_fasta(fasta_file, tag="real", mode="chunked", max_sequences=1000):
    os.makedirs("analysis", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + vocab
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

    # -------------------------------
    # Load sequences
    # -------------------------------
    if mode == "chunked":
        from preprocess_fasta import extract_chunks_from_fasta
        print("Extracting fixed-length chunks...")
        from preprocess_fasta import extract_chunks_from_fasta
        sequences = extract_chunks_from_fasta(fasta_file, chunk_size=cfg["chunk_size"], stride=cfg["stride"])
        sequences = sequences[:max_sequences]
    elif mode == "full":
        print("Reading full-length sequences...")
        records = list(SeqIO.parse(fasta_file, "fasta"))
        sequences = [str(r.seq).upper() for r in records if set(r.seq.upper()) <= set("ACGT")]
        sequences = sequences[:max_sequences]
    else:
        raise ValueError("Invalid mode. Use --mode chunked or full.")

    print(f"Loaded {len(sequences)} sequences.")

    # -------------------------------
    # Tokenize (truncate for model input)
    # -------------------------------
    print("Tokenizing...")
    vocab = build_kmer_vocab(cfg["k"])
    encoded_inputs = []
    true_lengths = []
    gc_vals = []

    for seq in sequences:
        if len(seq) < cfg["chunk_size"]:
            continue  # skip too short sequences
        true_lengths.append(len(seq))
        gc_vals.append(gc_content(seq))
        truncated_seq = seq[:cfg["chunk_size"]]  # trim to match model input
        tokens = tokenize_chunks([truncated_seq], cfg["k"], vocab)[0]
        encoded_inputs.append(tokens)

    print(f"Kept {len(encoded_inputs)} usable sequences for encoding.")

    # -------------------------------
    # Encode and project
    # -------------------------------
    latents = []
    with torch.no_grad():
        for tokens in tqdm(encoded_inputs, desc="Encoding"):
            x = torch.tensor(tokens).unsqueeze(0).to(device)
            mu, _ = model.encode(x)
            latents.append(mu.squeeze().cpu().numpy())
    latents = np.array(latents)

    reducer = umap.UMAP()
    projection = reducer.fit_transform(latents)

    # -------------------------------
    # Save metadata
    # -------------------------------
    df = pd.DataFrame({
        "gc": gc_vals,
        "length": true_lengths,
        "umap_1": projection[:, 0],
        "umap_2": projection[:, 1]
    })
    df.to_csv(f"analysis/{tag}_metadata.csv", index=False)

    # -------------------------------
    # Plots
    # -------------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="umap_1", y="umap_2", hue="gc", palette="viridis", s=20)
    plt.title(f"UMAP of {tag} sequences (colored by GC%)")
    plt.tight_layout()
    plt.savefig(f"analysis/{tag}_umap_gc.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["gc"], bins=40, kde=True)
    plt.title(f"GC Content of {tag} sequences")
    plt.xlabel("GC %")
    plt.tight_layout()
    plt.savefig(f"analysis/{tag}_gc_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["length"], bins=40)
    plt.title(f"Length Distribution of {tag} sequences")
    plt.xlabel("Length (bp)")
    plt.tight_layout()
    plt.savefig(f"analysis/{tag}_length_distribution.png")
    plt.close()

    # -------------------------------
    # Stats
    # -------------------------------
    stats = {
        "num_sequences": len(df),
        "mean_gc": float(np.mean(gc_vals)),
        "std_gc": float(np.std(gc_vals)),
        "mean_length": float(np.mean(true_lengths)),
        "std_length": float(np.std(true_lengths))
    }
    with open(f"analysis/{tag}_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"[✓] Analysis for '{tag}' complete. Saved to 'analysis/'.")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file path")
    parser.add_argument("--tag", type=str, default="real", help="Output tag name (used in filenames)")
    parser.add_argument("--mode", type=str, choices=["chunked", "full"], default="chunked",
                        help="Whether to extract fixed chunks or use full sequences")
    parser.add_argument("--max", type=int, default=1000, help="Max number of sequences to use")
    args = parser.parse_args()

    analyze_fasta(args.fasta, tag=args.tag, mode=args.mode, max_sequences=args.max)
