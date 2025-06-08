import torch
import json
import argparse
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import vae_model


# -------------------------------
# Reconstruct DNA from tokens
# -------------------------------
def tokens_to_dna(tokens, inv_vocab, k):
    kmers = [inv_vocab.get(str(tok), 'N' * k) for tok in tokens]
    seq = kmers[0]
    for kmer in kmers[1:]:
        seq += kmer[-1]
    return seq


# -------------------------------
# Sample from logits
# -------------------------------
def sample_softmax(logits):
    probs = torch.softmax(logits, dim=-1)  # [seq_len, vocab_size]
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [seq_len]
    return sampled_indices


# -------------------------------
# Load the lengths of the input sequences
# -------------------------------
def load_real_lengths(fasta_path):
    return [len(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")
            if "N" not in rec.seq and len(rec.seq) > 1000]


# -------------------------------
# Main Generation Function
# -------------------------------
def generate(num_genomes, input_fasta, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and vocab
    with open("saved/config.json") as f:
        cfg = json.load(f)
    with open("saved/vocab.json") as f:
        vocab = json.load(f)

    inv_vocab = {str(v): k for k, v in vocab.items()}
    k = cfg["k"]
    chunk_len = cfg["seq_len"] + k - 1

    # Load input length distribution
    real_lengths = load_real_lengths(input_fasta)
    print(f"Loaded {len(real_lengths)} real genome lengths for sampling.")

    # Load model
    model = vae_model.ConvVAE(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["seq_len"]
    ).to(device)
    model.load_state_dict(torch.load("saved/model.pt", map_location=device))
    model.eval()

    with open(output_path, "w") as f:
        for i in tqdm(range(num_genomes), desc="Generating genomes"):
            target_length = np.random.choice(real_lengths)  # Sample a length from the real genome lengths
            chunks_needed = (target_length + chunk_len - 1) // chunk_len

            dna_parts = []
            for _ in range(chunks_needed):
                with torch.no_grad():
                    z = torch.randn((1, cfg["latent_dim"])).to(device)
                    logits = model.decode(z).squeeze(0)  # [seq_len, vocab_size]
                    tokens = sample_softmax(logits).cpu().tolist()
                    chunk_dna = tokens_to_dna(tokens, inv_vocab, k)
                    dna_parts.append(chunk_dna)

            genome = ''.join(dna_parts)[:target_length]
            f.write(f">genome_{i}_length_{target_length}\n{genome}\n")

    print(f"[✓] Generated {num_genomes} variable-length genomes to {output_path}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_genomes", type=int, default=10, help="Number of synthetic genomes to generate")
    parser.add_argument("--input_fasta", type=str, default="real_genomes.fasta",
                        help="Input FASTA file for real genome lengths", required=True)
    parser.add_argument("--output", type=str, default="generated.fasta", help="Output FASTA file path")
    args = parser.parse_args()
    generate(args.num_genomes, args.input_fasta, args.output)
